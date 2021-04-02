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
        "capacity"              : 400000,
        "lr_ac"                 : 0.00005,
        "lr_cr"                 : 0.0005,
        "gamma"                 : 0.95,
        "first_hidden_layer"    : 128,
        "second_hidden_layer"   : 64,
        "tau"                   : 0.01,  # for target network soft update
        "delay_step"            : 6,
        "n_episode"             : 500000,
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
        "batch_size"            : 1024,
        "capacity"              : json_data["capacity"],
        "lr_ac"                 : json_data["lr_ac"],
        "lr_cr"                 : json_data["lr_cr"],
        "gamma"                 : 0.99,
        "first_hidden_layer"    : json_data["first_hidden_layer"],
        "second_hidden_layer"   : json_data["second_hidden_layer"],
        "tau"                   : json_data["tau"],  # for target network soft update
        "delay_step"            : 6,
        "n_episode"             : json_data["n_episode"],
        "episodes_before_train" : 100,
        "dim_laser"             : json_data["dim_laser"],
        "dim_laser_output"      : json_data["dim_laser_output"]
    })


##############################################################################################################
# MADDPG 
class MADDPG:
    def __init__(self, n_agents, dim_obs, dim_act, dim_laser, batch_size, las_output, first_hidden_layer, second_hidden_layer,
                 capacity, episodes_before_train, gamma, tau, lr_cr, lr_ac):

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

            # state_batch: batch_size x n_agents x dim_obs
            state_batch = torch.stack(batch.states).type(FloatTensor)
            laser_batch = torch.stack(batch.laser).type(FloatTensor)
            action_batch = torch.stack(batch.actions).type(FloatTensor)
            reward_batch = torch.stack(batch.rewards).type(FloatTensor)

            #######################################################
            # num_layer : number of recurrent layer, num_layer : 2 means 
            # stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results
            #
            # direction : bidirection lstm --> 2 / otherwise --> 1
            # 
            # batch.h_in : batch(tuple) x n_agents x num_output(tuple) x (num_layer * num_direction) x (dummy : 1)batch x hidden
            #
            # h_in_batch(torch type) : batch x n_agents x num_output x (num_layer * num_direction) x (dummy : 1)batch x hidden 
            h_in_list = list(batch.h_in)
            for b in range(len(h_in_list)):
                for a in range(self.n_agents):
                    if (type(h_in_list[b][a]) != torch.Tensor):
                        h_in_list[b][a] = torch.stack(h_in_list[b][a]).type(FloatTensor)
                h_in_list[b] = torch.stack(h_in_list[b]).type(FloatTensor)
            h_in_batch = torch.stack(h_in_list).type(FloatTensor)

            h_in_batch = h_in_batch.squeeze(4) # right h_in_batch : batch x n_agents x num_output x (num_layer * num_direction) x (dummy : 1)batch x hidden 
            h_in_batch = h_in_batch.transpose(0, 2) # right h_in_batch(torch type) : batch x n_agents x num_output x (num_layer * num_direction) x hidden 
            h_in_batch = h_in_batch.transpose(1, 3) # right h_in_batch(torch type) : num_output x n_agents x batch x (num_layer * num_direction) x hidden

            # h_in_batch(torch type) : num_output(h, c) x (num_layer * num_direction) x batch x n_agents x hidden 
            h_in_batch_agents = h_in_batch.contiguous().view(2, 1, self.batch_size, -1)

            # for h_out
            h_out_list = list(batch.h_out)
            non_final_h_out_batch = []
            for b in range(len(h_out_list)):
                if (h_out_list[b] != None):
                    non_final_h_out_batch.append(h_out_list[b])

            # non_final_h_out_batch : non_final_batch(tuple) x n_agents(list) x num_output(tuple) x (num_layer * num_direction) x (dummy : 1)batch x hidden
            # make full torch tensor
            for b in range(len(non_final_h_out_batch)):
                for a in range(self.n_agents):
                    if (type(non_final_h_out_batch[b][a]) != torch.Tensor):
                        non_final_h_out_batch[b][a] = torch.stack(non_final_h_out_batch[b][a]).type(FloatTensor)
                non_final_h_out_batch[b] = torch.stack(non_final_h_out_batch[b]).type(FloatTensor)
            h_out_batch = torch.stack(non_final_h_out_batch).type(FloatTensor)

            h_out_batch = h_out_batch.squeeze(4) # right h_out_batch : non_final_batch x n_agents x num_output x (num_layer * num_direction) x (dummy : 1)batch x hidden 
            h_out_batch = h_out_batch.transpose(0, 2) # right h_out_batch(torch type) : non_final_batch x n_agents x num_output x (num_layer * num_direction) x hidden 
            h_out_batch = h_out_batch.transpose(1, 3) # right h_out_batch(torch type) : num_output x n_agents x non_final_batch x (num_layer * num_direction) x hidden 
            
            # h_out_batch : num_output x (num_layer * num_direction) x non_final_batch x n_agnets x hidden
            h_out_batch_agents = h_out_batch.contiguous().view(2, 1, len(non_final_h_out_batch), -1)

            # for current agent
            whole_state = state_batch.view(self.batch_size, -1)
            whole_laser = laser_batch.view(self.batch_size, -1)
            whole_action = action_batch.view(self.batch_size, -1)

            # whole_h_in : num_output(tuple) x (num_layer * direction) x batch_size x hidden
            # whole_h_in = (h_in, c_in)
            whole_h_in = (h_in_batch_agents[0].detach(), h_in_batch_agents[1].detach())
            current_Q = self.critics[agent](whole_state, whole_action, whole_laser, whole_h_in)

            # : (batch_size_non_final) x n_agents x dim_obs
            non_final_next_states = torch.stack([s for s in batch.next_states if s is not None]).type(FloatTensor)
            non_final_next_lasers = torch.stack([l for l in batch.next_laser if l is not None]).type(FloatTensor)

            # n_agents x batch_size_non_final x next_action
            non_final_next_actions = [self.actors_target[i](non_final_next_states[:, i, :], non_final_next_lasers[:, i, :], (h_out_batch[0, :, :, i, :].contiguous(), h_out_batch[1, :, :, i, :].contiguous()))[0] for i in range(self.n_agents)]
            non_final_next_actions = torch.stack(non_final_next_actions)

            # batch_size_non_final x n_agents x next_action
            non_final_next_actions = (non_final_next_actions.transpose(0, 1).contiguous())

            target_Q = torch.zeros(self.batch_size).type(FloatTensor)
            target_Q[non_final_mask] = self.critics_target[agent](
                non_final_next_states.view(-1, self.n_agents * self.n_states),
                non_final_next_actions.view(-1, self.n_agents * self.n_actions),
                non_final_next_lasers.view(-1, self.n_agents * self.n_laser),
                (h_out_batch_agents[0].detach(), h_out_batch_agents[1].detach())
            ).squeeze()

            # TD Target = r + gamma * target_Q
            # TD Target shape : batch_size x 1 (agent)
            target_Q = (target_Q.unsqueeze(1) * self.GAMMA) + (reward_batch[:, agent].unsqueeze(1))

            # Update Critic Network
            loss_Q = nn.MSELoss()(current_Q, target_Q.detach())

            self.critic_optimizer[agent].zero_grad()
            loss_Q.backward()
            self.critic_optimizer[agent].step()

            # Update Actor Network
            state_i = state_batch[:, agent, :]
            laser_i = laser_batch[:, agent, :]

            action_i, _ = self.actors[agent](state_i, laser_i, (h_in_batch[0, :, :, agent, :].contiguous().detach(), h_in_batch[1, :, :, agent, :].contiguous().detach()))
            ac = action_batch.clone()
            ac[:, agent, :] = action_i
            whole_action = ac.view(self.batch_size, -1)

            # check replace true action(from buffer) to each agent's policy from obs --> make whole action from self.actor[agent](state_batch[:, agent, :])
            actor_loss = -self.critics[agent](whole_state, whole_action, whole_laser, whole_h_in).mean()
            
            # check performance
            # actor_loss += (action_i ** 2).mean() * 1e-3 # from openai reference code

            self.actor_optimizer[agent].zero_grad()
            actor_loss.backward()
            self.actor_optimizer[agent].step()

            c_loss.append(loss_Q)
            a_loss.append(actor_loss)

        for i in range(self.n_agents):
            soft_update(self.critics_target[i], self.critics[i], self.tau)
            soft_update(self.actors_target[i], self.actors[i], self.tau)

        return c_loss, a_loss

    # h_in_batch : n_agents x num_output(tuple) x (num_layer * num_direction) x batch x hidden
    def select_action(self, state_batch, laser_batch, h_in_batch):
        # state_batch: n_agents x state_dim
        actions = torch.zeros(
            self.n_agents,
            self.n_actions)
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        # agent x (num_layer x direction) x batch x hidden
        next_lstm_hidden_list = [0 for i in range(self.n_agents)]
        
        for i in range(self.n_agents):

            #  state_batch : nagents x state count
            sb = state_batch[i, :].detach()
            lb = laser_batch[i, :].detach()

            # h_in_batch : n_agents x num_output(tuple) x (num_layer * num_direction) x batch x hidden
            h_in = h_in_batch[i][0].detach()
            c_in = h_in_batch[i][1].detach()

            # h_in : num_output(tuple) x (num_layer * num_direction) x batch x hidden
            # lstm_hidden : num_output x (num_layer * num_direction) x batch x hidden
            act, lstm_hidden = self.actors[i](sb.unsqueeze(0), lb.unsqueeze(0), (h_in, c_in))
            act = act.squeeze(0)

            act += torch.from_numpy(
                np.random.randn(self.n_actions) * self.var[i]).type(FloatTensor)

            if self.episode_done > self.episodes_before_train and self.var[i] > 0.05:
                self.var[i] *= 0.999993
            act = torch.clamp(act, -1.0, 1.0)

            actions[i, :] = act

            if(type(lstm_hidden) != tuple):
                print("check")
            next_lstm_hidden_list[i] = lstm_hidden

        self.steps_done += 1

        # next_lstm_hidden_list : agent x num_output x (num_layer * num_direction) x batch x hidden_size
        return actions, next_lstm_hidden_list
##############################################################################################################
# MADDPG Replay Buffer
Experience = namedtuple('Experience',
                        ('states', 'laser', 'actions', 'next_states', 'next_laser', 'rewards', 'h_in', 'h_out'))

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
        self.n_agent = n_agent
        self.dim_observation = dim_observation
        self.dim_laser = dim_laser
        self.dim_action = dim_action
        obs_dim = dim_observation * n_agent
        act_dim = self.dim_action * n_agent
        las_dim = self.dim_laser * n_agent
        las_out_dim = las_output * n_agent

        hidden_dim = hidden_layer * n_agent
        second_hidden_dim = second_hidden_layer * n_agent

        self.lstm_hidden = hidden_layer * n_agent
        self.lstm_output = second_hidden_layer * n_agent

        self.PreFC = nn.Linear(las_dim, las_out_dim)
        self.FC1 = nn.Linear(obs_dim + las_out_dim + act_dim, hidden_dim)
        self.LSTM = nn.LSTM(hidden_dim, second_hidden_dim)
        self.FC3 = nn.Linear(second_hidden_dim, 1)

    # obs: batch_size * obs_dim
    def forward(self, obs, acts, laser, hidden):
        result = F.relu(self.PreFC(laser))
        combined = torch.cat([obs, result, acts], 1)
        result = F.relu(self.FC1(combined))

        # LSTM have to take input as (seq_len x batch x dim)
        result = result.view(1, -1, self.lstm_hidden)
        result, lstm_hidden = self.LSTM(result, hidden)

        # LSTM output : (seq_len x batch x dim)
        # --> (batch x dim)
        result = result.view(-1, self.lstm_output)
        return self.FC3(result)


class Actor(nn.Module):
    def __init__(self, dim_observation, dim_action, dim_laser, las_output = 10, hidden_layer=128, second_hidden_layer=64):
        super(Actor, self).__init__()
        self.FC1 = nn.Linear(dim_observation + las_output, hidden_layer)
        self.LSTM = nn.LSTM(hidden_layer, second_hidden_layer)
        self.FC3 = nn.Linear(second_hidden_layer, dim_action)

        self.lstm_hidden = hidden_layer
        self.lstm_output = second_hidden_layer
        self.PreFC = nn.Linear(dim_laser, las_output)

    def forward(self, obs, laser, hidden):
        result = F.relu(self.PreFC(laser))
        combined = torch.cat([obs, result], 1)
        result = F.relu(self.FC1(combined))

        # LSTM have to take input as (seq_len x batch x dim)
        result = result.view(1, -1, self.lstm_hidden)
        result, lstm_hidden = self.LSTM(result, hidden)

        # LSTM output : (seq_len x batch x dim)
        # --> (batch x dim) 
        result = result.view(-1, self.lstm_output)

        result = torch.tanh(self.FC3(result)) * 0.5
        return result, lstm_hidden

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
    rate = rospy.Rate(config.delay_step * 5 / 3.0)

    time.sleep(3)
    print("Start Training")

    FloatTensor = torch.cuda.FloatTensor if maddpg.use_cuda else torch.FloatTensor
    for i_episode in range(start_epi, config.n_episode):
        obs, laser = env.reset()
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        if isinstance(laser, np.ndarray):
            laser = torch.from_numpy(laser).float()

        # h_out_list : n_agents x num_output(tuple) x (num_layer * num_direction) x batch x hidden
        h_out_list = [0 for i in range(config.n_agents)]
        for i in range(len(h_out_list)):
            h_out_list[i] = (torch.zeros([1, 1, config.second_hidden_layer], dtype = torch.float, device = device), torch.zeros([1, 1, config.second_hidden_layer], dtype = torch.float, device = device))

        total_reward = 0.0
        n_step = 0

        past_obs_list = []
        past_laser_list= []
        past_action_list = []
        past_h_in_list = []
        past_h_out_list = []
        
        done = False

        while not done:
            h_in_list = h_out_list
            obs = obs.type(FloatTensor)
            laser = laser.type(FloatTensor)

            # h_out_list : n_agents x num_output(tuple) x (num_layer * num_direction) x batch x hidden
            torch_action, h_out_list = maddpg.select_action(obs, laser, h_in_list)
            action = torch_action.data.cpu()
            obs_, laser_, done, reward, _ = env.step(action.numpy().tolist())

            reward = torch.FloatTensor(reward).type(FloatTensor)
            obs_ = torch.from_numpy(obs_).float()
            laser_ = torch.from_numpy(laser_).float()

            store_h_in_list = [(h_in_list[0][0].data.cpu(), h_in_list[0][1].data.cpu()), (h_in_list[1][0].data.cpu(), h_in_list[1][1].data.cpu())]
            store_h_out_list = [(h_out_list[0][0].data.cpu(), h_out_list[0][1].data.cpu()), (h_out_list[1][0].data.cpu(), h_out_list[1][1].data.cpu())]

            if done:
                next_obs = None
                next_laser = None
                store_h_out_list = None
            else:
                next_obs = obs_
                next_laser = laser_

            total_reward += reward.sum()
            if (n_step < config.delay_step):
                past_obs_list.append(obs.data.cpu())
                past_laser_list.append(laser.data.cpu())
                past_action_list.append(action)
                past_h_in_list.append(store_h_in_list)
                past_h_out_list.append(store_h_out_list)
            elif (done):
                maddpg.memory.push(past_obs_list.pop(0), past_laser_list.pop(0), past_action_list.pop(0), next_obs, next_laser, reward, past_h_in_list.pop(0), None)
                past_obs_list.append(obs.data.cpu())
                past_laser_list.append(laser.data.cpu())
                past_action_list.append(action)
                past_h_in_list.append(store_h_in_list)
                past_h_out_list.append(store_h_out_list)
            else:
                maddpg.memory.push(past_obs_list.pop(0), past_laser_list.pop(0), past_action_list.pop(0), next_obs, next_laser, reward, past_h_in_list.pop(0), past_h_out_list.pop(0))
                past_obs_list.append(obs.data.cpu())
                past_laser_list.append(laser.data.cpu())
                past_action_list.append(action)
                past_h_in_list.append(store_h_in_list)
                past_h_out_list.append(store_h_out_list)

            n_step += 1

            obs = next_obs
            laser = next_laser
            rate.sleep()

        if(args.test == False):
            for i in range(10):
                c_loss, a_loss = maddpg.update_policy()
            
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
                model_path = os.path.join("/home/dwkim/RL_ws/src/rl/src", Utils.Save_path)
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
                        path = '/home/dwkim/.ros/sitl_iris_' + str(i) + '/log'
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