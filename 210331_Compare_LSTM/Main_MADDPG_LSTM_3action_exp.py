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
#Hyperparameters
if (args.model_path == None):
    config = EasyDict({
        "start_epi"             : 0,
        "n_agents"              : 2,
        "n_targets"             : 1,
        "dim_obs"               : 12,
        "dim_act"               : 3,
        "batch_size"            : 50, # should be available divcde by 10
        "capacity"              : 1000,
        "trajectory_time"       : 20,
        "lr_ac"                 : 0.00001,
        "lr_cr"                 : 0.0005,
        "gamma"                 : 0.95,
        "first_hidden_layer"    : 128,
        "second_hidden_layer"   : 128,
        "tau"                   : 0.01,  # for target network soft update
        "delay_step"            : 6,
        "n_episode"             : 50000,
        "episodes_before_train" : 100,
        "dim_laser"             : 1000,
        "dim_laser_output"      : 100
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
        "trajectory_time"       : json_data["trajectory_time"],
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

#############################################################################################################
# CUDA
if(torch.cuda.is_available()):
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

##############################################################################################################
# MADDPG 
class MADDPG:
    def __init__(self, 
                n_agents, 
                dim_obs, dim_act, dim_laser, 
                batch_size, 
                laser_output, 
                first_hidden_layer, second_hidden_layer,
                capacity, 
                trajectory_time, 
                episodes_before_train, 
                gamma, tau, lr_cr, lr_ac):

        self.actors = [Actor(dim_obs, dim_act, dim_laser, laser_output, first_hidden_layer, second_hidden_layer) for i in range(n_agents)]
        self.critics = [Critic(n_agents, dim_obs, dim_act, dim_laser, laser_output, first_hidden_layer, second_hidden_layer) for i in range(n_agents)]

        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = dim_act
        self.n_laser = dim_laser
        self.memory = ReplayMemory(capacity, trajectory_time)
        self.batch_size = batch_size
        self.trajectory_time = trajectory_time
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

        loss_Q = [0, 0]
        actor_loss = [0, 0]

        slice_cnt = 10

        c_loss = []
        a_loss = []
        # trajectorys : batch x trajectory
        # trajectory : [Exp0, ... ,ExpS-1]
        trajectorys = self.memory.sample(int(self.batch_size/slice_cnt))

        # slice_transition = [Exp_0(s=, a=), Exp_1(s=, a=), ... , Exp_t(s=, a=)]
        # slice_transitions = [slice_transition_0, slice_transition_1, ..., slice_transition_99]
        # slice_transitions : batch_size x trajectroy_time x Experience
        slice_transitions = []
        for trajectory in trajectorys: # len = self.batch_size ( = 100 )
            total_len = len(trajectory) # total_len = episode_lenght
            for i in range(slice_cnt):
                index = random.randint(0, total_len - self.trajectory_time) # check done transition
                slice_transitions.append(trajectory[index:index + self.trajectory_time])
        
        # np_batch : batch x trajectory_time x Exp_cnt (= 7 : s, l, a, s', l', r, d)
        np_batch = np.array(slice_transitions)
        np_batch = np.transpose(np_batch, (1, 0, 2))

        # state_batch : trajectory_time x batch_size x  Episode_element x agent x element
        exp_state_batch = np.zeros((self.trajectory_time, self.batch_size, self.n_agents, self.n_states))
        exp_laser_batch = np.zeros((self.trajectory_time, self.batch_size, self.n_agents, self.n_laser))
        exp_action_batch = np.zeros((self.trajectory_time, self.batch_size, self.n_agents, self.n_actions))
        exp_next_state_batch = np.zeros((self.trajectory_time, self.batch_size, self.n_agents, self.n_states))
        exp_next_laser_batch = np.zeros((self.trajectory_time, self.batch_size, self.n_agents, self.n_laser))
        exp_reward_batch = np.zeros((self.trajectory_time, self.batch_size, self.n_agents))
        exp_non_final_mask = np.zeros((self.trajectory_time, self.batch_size, 1))
        for t in range(self.trajectory_time):
            for b in range(self.batch_size):
                exp_state_batch[t, b] = np_batch[t, b, 0]
                exp_laser_batch[t, b] = np_batch[t, b, 1]
                exp_action_batch[t, b] = np_batch[t, b, 2]
                exp_next_state_batch[t, b] = np_batch[t, b, 3]
                exp_next_laser_batch[t, b] = np_batch[t, b, 4]
                exp_reward_batch[t, b] = np_batch[t, b, 5]
                exp_non_final_mask[t, b] = 0.0 if np_batch[t, b, 6] == True else 1.0
        
        exp_state_batch = torch.from_numpy(exp_state_batch)
        exp_laser_batch = torch.from_numpy(exp_laser_batch)
        exp_action_batch = torch.from_numpy(exp_action_batch)
        exp_next_state_batch = torch.from_numpy(exp_next_state_batch)
        exp_next_laser_batch = torch.from_numpy(exp_next_laser_batch)
        exp_reward_batch = torch.from_numpy(exp_reward_batch)
        exp_non_final_mask = torch.from_numpy(exp_non_final_mask)

        initial_M_hidden = (torch.zeros([1, self.batch_size, config.second_hidden_layer], dtype = torch.float, device = device), torch.zeros([1, self.batch_size, config.second_hidden_layer], dtype = torch.float, device = device))
        target_initial_M_hidden = (torch.zeros([1, self.batch_size, config.second_hidden_layer], dtype = torch.float, device = device), torch.zeros([1, self.batch_size, config.second_hidden_layer], dtype = torch.float, device = device))

        h = [initial_M_hidden, initial_M_hidden]
        target_h = [target_initial_M_hidden, target_initial_M_hidden]

        for i in range(self.trajectory_time):
            # state_batch: batch_size x n_agents x dim_obs
            state_batch = exp_state_batch[i].type(FloatTensor)
            laser_batch = exp_laser_batch[i].type(FloatTensor)
            action_batch = exp_action_batch[i].type(FloatTensor)
            next_state_batch = exp_next_state_batch[i].type(FloatTensor)
            next_laser_batch = exp_next_laser_batch[i].type(FloatTensor)
            reward_batch = exp_reward_batch[i].type(FloatTensor)

            # non_final_mask : batch x 1
            non_final_mask = exp_non_final_mask[i].type(FloatTensor)

            # for current agent
            whole_state = state_batch.view(self.batch_size, -1)
            whole_laser = laser_batch.view(self.batch_size, -1)
            whole_action = action_batch.view(self.batch_size, -1)

            # h : agents(list) x num_hidden(tuple) x (torch)seq_len x batch_size x hidden
            # desired whole_hc : (tuple)num_hidden x (torch)seq_len x batch_size, hidden
            whole_h = torch.stack([h[0][0], h[1][0]]).view(1, self.batch_size, -1)
            whole_c = torch.stack([h[0][1], h[1][1]]).view(1, self.batch_size, -1)
            whole_hc = (whole_h, whole_c)

            # calculate Next Action
            action_target_list = [self.actors_target[i](next_state_batch[:, i, :], next_laser_batch[:, i, :], target_h[i]) for i in range(self.n_agents)]
            next_action_batch, target_h = [], []
            for action, target in action_target_list:
                next_action_batch.append(action)
                target_h.append(target)

            # next_action_batch : n_agents x batch_size x next_action
            next_action_batch = torch.stack(next_action_batch)
            next_action_batch = (next_action_batch.transpose(0, 1).contiguous())

            # target_h : n_agent x seq_len x batch_size x hidden
            whole_target_h = torch.stack([target_h[0][0], target_h[1][0]]).view(1, self.batch_size, -1)
            whole_target_c = torch.stack([target_h[0][1], target_h[1][1]]).view(1, self.batch_size, -1)
            whole_target_hc = (whole_target_h, whole_target_c)

            for agent in range(self.n_agents):
                current_Q = self.critics[agent](whole_state, whole_action, whole_laser, whole_hc)

                target_Q = self.critics_target[agent](
                    next_state_batch.view(-1, self.n_agents * self.n_states),
                    next_action_batch.view(-1, self.n_agents * self.n_actions),
                    next_laser_batch.view(-1, self.n_agents * self.n_laser),
                    (whole_target_hc[0].detach(), whole_target_hc[1].detach())
                ).squeeze()

                # TD Target shape : batch_size x 1 (agent)
                target_Q = (non_final_mask.unsqueeze(1) * target_Q.unsqueeze(1) * self.GAMMA) + (reward_batch[:, agent].unsqueeze(1))

                # Update Critic Network
                loss_Q[agent] += nn.MSELoss()(current_Q, target_Q.detach()) / self.trajectory_time

                # Update Actor Network
                state_i = state_batch[:, agent, :]
                laser_i = laser_batch[:, agent, :]

                action_i, h[agent] = self.actors[agent](state_i, laser_i, (h[agent][0].detach(), h[agent][1].detach()))
                ac = action_batch.clone()
                ac[:, agent, :] = action_i
                whole_action = ac.view(self.batch_size, -1)

                # check replace true action(from buffer) to each agent's policy from obs --> make whole action from self.actor[agent](state_batch[:, agent, :])
                actor_loss[agent] += -self.critics[agent](whole_state, whole_action, whole_laser, whole_hc).mean() / self.trajectory_time

        for agent in range(self.n_agents):
            self.critic_optimizer[agent].zero_grad()
            loss_Q[agent].backward(retain_graph = True)
            self.critic_optimizer[agent].step()

            self.actor_optimizer[agent].zero_grad()
            actor_loss[agent].backward(retain_graph = True)
            self.actor_optimizer[agent].step()

            c_loss.append(loss_Q[agent])
            a_loss.append(actor_loss[agent])

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
                        ('states', 'laser', 'actions', 'next_states', 'next_laser', 'rewards', 'done'))

class ReplayMemory:
    def __init__(self, capacity, trajectory_time):
        self.trajectory_time = trajectory_time
        self.capacity = capacity
        self.trajectory = []
        self.memory = []
        self.position = 0

    def push(self, *args):
        self.trajectory.append(Experience(*args))
            
    def update(self):
        if (len(self.trajectory) > 100):
            if (len(self.memory) < self.capacity):
                self.memory.append(None)

            self.memory[self.position] = self.trajectory
            self.position = (self.position + 1) % self.capacity
        self.trajectory = []

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
                    config.dim_obs, config.dim_act, config.dim_laser, 
                    config.batch_size, 
                    config.dim_laser_output, 
                    config.first_hidden_layer, config.second_hidden_layer, 
                    config.capacity, 
                    config.trajectory_time, 
                    config.episodes_before_train, 
                    config.gamma, config.tau, config.lr_cr, config.lr_ac)

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
    rate = rospy.Rate(config.delay_step * 5 / 3)

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

        past_obs_list, past_laser_list, past_action_list = [], [], []
        
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

            total_reward += reward.sum()

            if (n_step >= config.delay_step):
                maddpg.memory.push(past_obs_list.pop(0), past_laser_list.pop(0), past_action_list.pop(0), 
                                    obs_.cpu().numpy(), laser_.cpu().numpy(), reward.cpu(), done)

            past_obs_list.append(obs.data.cpu().numpy())
            past_laser_list.append(laser.data.cpu().numpy())
            past_action_list.append(action)

            n_step += 1

            obs = obs_
            laser = laser_
            rate.sleep()

        maddpg.memory.update()

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