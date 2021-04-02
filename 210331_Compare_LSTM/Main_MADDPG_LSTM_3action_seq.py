#!/usr/bin/env python
# ROS
import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Float32MultiArray, Int32
from nav_msgs.msg import Odometry
import tf
import shutil

# Env
import os
import datetime
import numpy as np
import random, collections
import math
import time
import argparse
from time import sleep

# RL training
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions import Normal
from collections import namedtuple

# Parameter
from easydict import EasyDict
import json

# Tensorboard
from torch.utils.tensorboard import SummaryWriter

import Utils
from Env_3action import Gazebo_Env
from Train_Utils import soft_update, hard_update, OrnsteinUhlenbeckProcess


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--test', type=bool, default=False)
args = parser.parse_args()

pc_name = 'dwkim'

##############################################################################
if(torch.cuda.is_available()):
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

##############################################################################
#Hyperparameters
if (args.model_path == None):
    config = EasyDict({
        "start_epi"             : 0,
        "n_agents"              : 2,
        "n_targets"             : 1,
        "dim_obs"               : 12,
        "dim_act"               : 3,
        "batch_size"            : 1024,
        "capacity"              : 50000,
        "time_step"             : 10,
        "lr_ac"                 : 0.00001,
        "lr_cr"                 : 0.0005,
        "gamma"                 : 0.95,
        "first_hidden_layer"    : 128,
        "second_hidden_layer"   : 64,
        "tau"                   : 0.01,  # for target network soft update
        "delay_step"            : 6,
        "n_episode"             : 50000,
        "episodes_before_train" : 100,
        "dim_laser"             : 1000,
        "dim_laser_output"      : 10
    })
else:
    json_path = args.model_path + "/config.json"
    json_file = open(json_path)
    json_data = json.load(json_file)
    
    config = EasyDict({
        "start_epi"             : json_data["start_epi"],
        "n_agents"              : json_data["n_agents"],
        "n_targets"             : json_data["n_targets"],
        "dim_obs"               : json_data["dim_obs"],
        "dim_act"               : json_data["dim_act"],
        "batch_size"            : json_data["batch_size"],
        "capacity"              : json_data["capacity"],
        "time_step"             : json_data["time_step"],
        "lr_ac"                 : json_data["lr_ac"],
        "lr_cr"                 : json_data["lr_cr"],
        "gamma"                 : json_data["gamma"],
        "first_hidden_layer"    : json_data["first_hidden_layer"],
        "second_hidden_layer"   : json_data["second_hidden_layer"],
        "tau"                   : json_data["tau"],  # for target network soft update
        "delay_step"            : json_data["delay_step"],
        "n_episode"             : json_data["n_episode"],
        "episodes_before_train" : json_data["episodes_before_train"],
        "dim_laser"             : json_data["dim_laser"],
        "dim_laser_output"      : json_data["dim_laser_output"]
    })


##############################################################################################################
# MADDPG 
class MADDPG:
    def __init__(self, 
                n_agents, 
                dim_obs, dim_act, dim_laser, 
                batch_size, 
                las_output, 
                first_hidden_layer, 
                second_hidden_layer,
                capacity, 
                time_step, 
                episodes_before_train, 
                gamma, tau, lr_cr, lr_ac):

        self.actors = [Actor(dim_obs, dim_act, dim_laser, las_output, first_hidden_layer, second_hidden_layer) for i in range(n_agents)]
        self.critics = [Critic(n_agents, dim_obs, dim_act, dim_laser, las_output, first_hidden_layer, second_hidden_layer) for i in range(n_agents)]

        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = dim_act
        self.n_laser = dim_laser
        self.memory = ReplayMemory(capacity)
        self.batch_size = batch_size
        self.time_step = time_step
        self.use_cuda = torch.cuda.is_available()
        self.episodes_before_train = episodes_before_train

        self.GAMMA = gamma
        self.tau = tau

        if(args.test == False):
            self.var = [1.0 for i in range(n_agents)]
        else:
            self.var = [0.0 for i in range(n_agents)]

        self.critic_optimizer = [optim.Adam(x.parameters(), lr=lr_cr) for x in self.critics]
        self.actor_optimizer = [optim.Adam(x.parameters(), lr=lr_ac) for x in self.actors]

        if self.use_cuda:
            for x in self.actors:
                x.cuda()
            for x in self.critics:
                x.cuda()
            for x in self.actors_target:
                x.cuda()
            for x in self.critics_target:
                x.cuda()

        self.steps_done = 0
        self.episode_done = 0

    def update_policy(self):
        # do not train until exploration is enough
        if self.episode_done <= self.episodes_before_train:
            return None, None

        BoolTensor = torch.cuda.BoolTensor if self.use_cuda else torch.BoolTensor
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        c_loss = []
        a_loss = []
        for agent in range(self.n_agents):
            transitions = self.memory.sample(self.batch_size)
            batch = Experience(*zip(*transitions))
            non_final_mask = BoolTensor(list(map(lambda s: s is not None, batch.next_states)))

            #######################################################################
            # CURRENT state_batch : batch_size x time_step x n_agents x dim_obs
            # DESIRED state_batch : time_step x batch_size x n_agents x dim_obs
            #
            # reward_batch : batch_size x n_agents
            state_batch = torch.stack(batch.states).type(FloatTensor).transpose(1, 0)
            laser_batch = torch.stack(batch.laser).type(FloatTensor).transpose(1, 0)
            action_batch = torch.stack(batch.actions).type(FloatTensor).transpose(1, 0)
            reward_batch = torch.stack(batch.rewards).type(FloatTensor)
            next_state_batch = torch.stack(batch.next_states).type(FloatTensor).transpose(1, 0)
            next_laser_batch = torch.stack(batch.next_laser).type(FloatTensor).transpose(1, 0)
            non_final_mask = ~torch.tensor(batch.done).type(BoolTensor)

            #######################################################################
            # DESIRED whole_state_batch : time_step x batch_size x dim_obs
            whole_state = state_batch.view(self.time_step, self.batch_size, -1)
            whole_laser = laser_batch.view(self.time_step, self.batch_size, -1)
            whole_action = action_batch.view(self.time_step, self.batch_size, -1)

            # critis output shape : time_step x batch_size x value
            current_Q = self.critics[agent](whole_state, whole_action, whole_laser, config.batch_size)[-1]

            # agent x time_step x batch_size x n_action
            next_action_batch = [self.actors_target[i](next_state_batch[:, :, i, :], next_laser_batch[:, :, i, :], config.batch_size) for i in range(self.n_agents)]
            
            # non_final_next_actions : time_step x batch_size x n_agents x n_action
            next_action_batch = torch.stack(next_action_batch).transpose(0, 2).transpose(0, 1).contiguous() 

            target_Q = torch.zeros(self.batch_size).type(FloatTensor)
            target_Q = self.critics_target[agent](
                next_state_batch.view(self.time_step, -1, self.n_agents * self.n_states),
                next_action_batch.view(self.time_step, -1, self.n_agents * self.n_actions),
                next_laser_batch.view(self.time_step, -1, self.n_agents * self.n_laser),
                config.batch_size)[-1].squeeze()

            # TD Target = r + gamma * target_Q
            # TD Target shape : batch_size x 1 (agent)
            target_Q = (non_final_mask.unsqueeze(1) * target_Q.unsqueeze(1) * self.GAMMA) + (reward_batch[:, agent].unsqueeze(1))

            # Update Critic Network
            loss_Q = nn.MSELoss()(current_Q, target_Q.detach())

            self.critic_optimizer[agent].zero_grad()
            loss_Q.backward()
            self.critic_optimizer[agent].step()

            # Update Actor Network
            state_i = state_batch[:, :, agent, :]
            laser_i = laser_batch[:, :, agent, :]

            # time x batch x dim_act
            action_i = self.actors[agent](state_i, laser_i, config.batch_size)
            ac = action_batch.clone()
            ac[:, :, agent, :] = action_i
            whole_action = ac.view(self.time_step, self.batch_size, -1)

            actor_loss = -self.critics[agent](whole_state, whole_action, whole_laser, config.batch_size)[-1].mean()
            
            self.actor_optimizer[agent].zero_grad()
            actor_loss.backward()
            self.actor_optimizer[agent].step()

            c_loss.append(loss_Q)
            a_loss.append(actor_loss)

        for i in range(self.n_agents):
            soft_update(self.critics_target[i], self.critics[i], self.tau)
            soft_update(self.actors_target[i], self.actors[i], self.tau)

        return c_loss, a_loss

    def select_action(self, state_batch, laser_batch):
        # state_batch: time_step x n_agents x state_dim
        actions = torch.zeros(
            self.n_agents,
            self.n_actions)
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        for i in range(self.n_agents):

            #  state_batch : time_step x nagents x n_state
            sb = state_batch[:, i, :].detach()
            lb = laser_batch[:, i, :].detach()

            act = self.actors[i](sb.unsqueeze(1), lb.unsqueeze(1), 1) # act : time_step x batch_size x n_action
            act = act[-1] # act : batch_size x n_action

            act += torch.from_numpy(
                np.random.randn(self.n_actions) * self.var[i]).type(FloatTensor)

            if self.episode_done > self.episodes_before_train and self.var[i] > 0.05:
                self.var[i] *= 0.999993
            act = torch.clamp(act, -1.0, 1.0)

            actions[i, :] = act

        self.steps_done += 1

        # final_action : n_agents x n_action
        return actions
##############################################################################################################
# MADDPG Replay Buffer
Experience = namedtuple('Experience',
                        ('states', 'laser', 'actions', 'next_states', 'next_laser', 'rewards', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

##############################################################################################################
# MADDPG Model
class Critic(nn.Module):
    def __init__(self, n_agent, dim_observation, dim_action, dim_laser, las_output = 10, hidden_layer=128, second_hidden_layer=64):
        super(Critic, self).__init__()
        obs_dim = dim_observation * n_agent
        act_dim = dim_action * n_agent
        las_dim = dim_laser * n_agent
        las_out_dim = las_output * n_agent
        hidden_dim = hidden_layer * n_agent
        second_hidden_dim = second_hidden_layer * n_agent
        self.lstm_output = second_hidden_layer * n_agent
        
        self.PreFC = nn.Linear(las_dim, las_out_dim)
        self.FC1 = nn.Linear(obs_dim + las_out_dim + act_dim, hidden_dim)
        self.LSTM = nn.LSTM(hidden_dim, second_hidden_dim, config.time_step)
        self.FC3 = nn.Linear(second_hidden_dim, 1)

    # obs: batch_size * obs_dim
    def forward(self, obs, acts, laser, batch):
        #############################################
        # obs : time_step x batch_size x n_state
        # laser : time_step x batch_size x n_laser
        # acts : time_step x batch_size x n_action
        #############################################
        hidden = (torch.zeros((config.time_step, batch, self.lstm_output), device = device), torch.zeros((config.time_step, batch, self.lstm_output), device = device))

        result = F.relu(self.PreFC(laser))

        combined = torch.cat([obs, result, acts], 2)
        result = F.relu(self.FC1(combined))
        result, lstm_hidden = self.LSTM(result, hidden)

        # final return value : time_step x batch_size x 1
        return self.FC3(result)


class Actor(nn.Module):
    def __init__(self, dim_observation, dim_action, dim_laser, las_output = 10, hidden_layer=128, second_hidden_layer=64):
        super(Actor, self).__init__()
        self.lstm_output = second_hidden_layer

        self.FC1 = nn.Linear(dim_observation + las_output, hidden_layer)
        self.LSTM = nn.LSTM(hidden_layer, second_hidden_layer, config.time_step)
        self.FC3 = nn.Linear(second_hidden_layer, dim_action)
        self.PreFC = nn.Linear(dim_laser, las_output)

    def forward(self, obs, laser, batch):
        #############################################
        # obs : time_step x batch_size x n_state
        # laser : time_step x batch_size x n_laser
        #############################################
        hidden = (torch.zeros((config.time_step, batch, self.lstm_output), device = device), torch.zeros((config.time_step, batch, self.lstm_output), device = device))

        result = F.relu(self.PreFC(laser))
        combined = torch.cat([obs, result], 2)
        result = F.relu(self.FC1(combined))

        result, lstm_hidden = self.LSTM(result, hidden)
        result = torch.tanh(self.FC3(result)) * 0.5 

        # final result shape : time_step x batch_size x n_action
        return result

##################################################################################
def main():
    env = Gazebo_Env(config.n_agents, config.n_targets, config.dim_laser)
    reward_record = []

    print_interval = 100
    score = 0

    maddpg = MADDPG(config.n_agents, 
                    config.dim_obs, 
                    config.dim_act, 
                    config.dim_laser, 
                    config.batch_size, 
                    config.dim_laser_output, 
                    config.first_hidden_layer, 
                    config.second_hidden_layer, 
                    config.capacity, 
                    config.time_step, 
                    config.episodes_before_train, 
                    config.gamma, 
                    config.tau, 
                    config.lr_cr, 
                    config.lr_ac)

    if (args.model_path != None):
        file_list = os.listdir(args.model_path)

        max_index = 0
        max_value = 0
        for idx in range(len(file_list)):
            if(Utils.isint(file_list[idx][5:-4])):
                if(max_value < int(file_list[idx][5:-4])):
                    max_index = idx
                    max_value = int(file_list[idx][5:-4])
        last_file = file_list[max_index]

        path = args.model_path + '/' + last_file
        print(path)

        checkpoint = torch.load(path)
        start_epi = checkpoint['n_epi']
        for a, aopt, params, opt in zip(maddpg.actors, maddpg.actor_optimizer, checkpoint['actor_params'], checkpoint['actor_optim']):
            a.load_state_dict(params)
            aopt.load_state_dict(opt)
        for a, aopt, params, opt in zip(maddpg.critics, maddpg.critic_optimizer, checkpoint['critic_params'], checkpoint['critic_optim']):
            a.load_state_dict(params)
            aopt.load_state_dict(opt)
    else:
        start_epi = config.start_epi

    # tensorboard --logdir=runs
    writer = SummaryWriter()
    rate = rospy.Rate(10)

    time.sleep(3)
    print("Start Training")

    FloatTensor = torch.cuda.FloatTensor if maddpg.use_cuda else torch.FloatTensor
    for i_episode in range(start_epi, config.n_episode):
        obs, laser = env.reset()
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        if isinstance(laser, np.ndarray):
            laser = torch.from_numpy(laser).float()

        total_reward = 0.0
        n_step = 0

        past_obs_list, past_laser_list, past_action_list = [], [], []
        done = False

        # obs_list : time_step x n_agent x n_state
        obs_list = [obs.type(FloatTensor) for i in range(config.time_step)]
        laser_list = [laser.type(FloatTensor) for i in range(config.time_step)]
        action_list = [torch.zeros((config.n_agents, config.dim_act)) for i in range(config.time_step)]

        while not done:
            torch_action = maddpg.select_action(torch.stack(obs_list), torch.stack(laser_list))
            action = torch_action.data.cpu()
            obs_, laser_, done, reward, _ = env.step(action.numpy().tolist())

            reward = torch.FloatTensor(reward).type(FloatTensor)
            obs_ = torch.from_numpy(obs_).type(FloatTensor)
            laser_ = torch.from_numpy(laser_).type(FloatTensor)

            #############################
            if (n_step >= config.time_step):
                past_obs_list.append([obs.data.cpu() for obs in obs_list])
                past_laser_list.append([laser.data.cpu() for laser in laser_list])
                past_action_list.append(action_list)
            
            #############################
            # create state list
            obs_list.pop(0)
            obs_list.append(obs_)

            laser_list.pop(0)
            laser_list.append(laser_)

            action_list.pop(0)
            action_list.append(action)

            ############################
            total_reward += reward.sum()

            if (n_step >= config.time_step + config.delay_step):
                maddpg.memory.push(
                    torch.stack(past_obs_list.pop(0)),
                    torch.stack(past_laser_list.pop(0)),
                    torch.stack(past_action_list.pop(0)),
                    torch.stack([obs_.data.cpu() for obs_ in obs_list]),
                    torch.stack([laser_.data.cpu() for laser_ in laser_list]),
                    reward,
                    done
                )
                
            n_step += 1
            rate.sleep()

        if(args.test == False):
            for i in range(10):
                c_loss, a_loss = maddpg.update_policy()

        print(len(maddpg.memory))
        score += total_reward
        maddpg.episode_done += 1
        reward_record.append(total_reward)

        if (args.test == False and c_loss != None):
            writer.add_scalar("Total Reward", total_reward, i_episode)

            writer.add_scalar("Agent0 Critic Loss", c_loss[0].detach(), i_episode)
            writer.add_scalar("Agent1 Critic Loss", c_loss[1].detach(), i_episode)

            writer.add_scalar("Agent0 Actor Loss", a_loss[0].detach(), i_episode)
            writer.add_scalar("Agent1 Actor Loss", a_loss[1].detach(), i_episode)
            
        print("Episode : " + str(i_episode) + " / Reward : " + str(total_reward))
        print(" ")

        if (args.test == False):
            if i_episode == 0 or i_episode == start_epi:
                model_path = os.path.join("/home/" + pc_name + "/RL_ws/src/rl/src", Utils.Save_path)
                if not os.path.isdir(model_path):
                    os.makedirs(model_path)
                
                with open(model_path + '/config.json', 'w') as f:
                    json.dump(vars(config), f)

            if i_episode % print_interval == 0 and (i_episode != 0 and i_episode != start_epi):
                #####################################
                try:
                    path = '/tmp/'
                    for filename in os.listdir(path):
                        if(filename[0] == 'F'):
                            remove_path = path + filename
                            os.remove(remove_path)
                    print("Delete tmp Success")
                except:
                    print("No tmp file")
                #####################################
                try:
                    for i in range(config.n_agents + config.n_targets):
                        path = '/home/' + pc_name + '/.ros/sitl_iris_' + str(i) + '/log'
                        shutil.rmtree(path)
                    print("Delete log file")
                except:
                    print("No log file")
                #####################################
                avg_score_str = str(int(score/print_interval * 1000))
                ckpt_path = os.path.join(model_path, 'ckpt_'+ avg_score_str + '.pth')

                torch.save(
                    {
                    'n_epi' : i_episode,
                    'actor_params' : [a.state_dict() for a in maddpg.actors],
                    'critic_params' : [a.state_dict() for a in maddpg.critics],
                    'actor_optim' : [a.state_dict() for a in maddpg.actor_optimizer],
                    'critic_optim' : [a.state_dict() for a in maddpg.critic_optimizer],
                    }, ckpt_path)

                print("# of episode :{}, avg score : {:.1f}".format(i_episode, score/print_interval))
                score = 0.0
        
if __name__ == '__main__':
    main()