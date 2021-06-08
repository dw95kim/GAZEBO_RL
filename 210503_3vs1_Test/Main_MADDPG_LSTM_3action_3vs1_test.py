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
# from time import sleep
from itertools import permutations

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
from Env_3action_test import Gazebo_Env
from Train_Utils import soft_update, hard_update, OrnsteinUhlenbeckProcess


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default=None)
args = parser.parse_args()

pc_name = "usrg"
max_cmd = 0.5 # tracking dron's max velocity

if(torch.cuda.is_available()):
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

##############################################################################
#Hyperparameters
if (args.model_path == None):
    config = EasyDict({
        "start_epi"             : 0,
        "n_agents"              : 3,
        "n_targets"             : 1,
        "dim_obs"               : 12,
        "dim_act"               : 3,
        "delay_step"            : 6,
        "n_episode"             : 50000,
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
        "delay_step"            : json_data["delay_step"],
        "n_episode"             : json_data["n_episode"],
        "dim_laser"             : json_data["dim_laser"],
        "dim_laser_output"      : json_data["dim_laser_output"]
    })


##############################################################################################################
# MADDPG 
class MADDPG:
    def __init__(self, 
                n_agents, 
                dim_obs, dim_act, dim_laser, dim_las_output,
                first_hidden_layer, second_hidden_layer):

        self.actors = [Actor(dim_obs, dim_act, dim_laser, dim_las_output, first_hidden_layer, second_hidden_layer) for i in range(n_agents)]
        self.critics = [Critic(n_agents, dim_obs, dim_act, dim_laser, dim_las_output, first_hidden_layer, second_hidden_layer) for i in range(n_agents)]

        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = dim_act
        self.n_laser = dim_laser
        self.n_laser_out = dim_las_output
        self.use_cuda = torch.cuda.is_available()

        for a, c, at, ct in zip(self.actors, self.critics, self.actors_target, self.critics_target):
            a.to(device)
            c.to(device)
            at.to(device)
            ct.to(device)

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
            act = torch.clamp(act, -1.0, 1.0)

            actions[i, :] = act

            next_lstm_hidden_list[i] = lstm_hidden

        # next_lstm_hidden_list : agent x num_output x (num_layer * num_direction) x batch x hidden_size
        return actions, next_lstm_hidden_list

##############################################################################################################
# MADDPG Model
class Critic(nn.Module):
    def __init__(self, n_agent, dim_observation, dim_action, dim_laser, las_output, hidden_layer, second_hidden_layer):
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
    def __init__(self, dim_observation, dim_action, dim_laser, las_output, hidden_layer, second_hidden_layer):
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

        result = torch.tanh(self.FC3(result)) * max_cmd
        return result, lstm_hidden

##################################################################################
def main():
    env = Gazebo_Env(config.n_agents, config.n_targets, config.dim_laser)
    reward_record = []

    print_interval = 100
    score = 0

    maddpg = MADDPG(config.n_agents, 
                    config.dim_obs, config.dim_act, config.dim_laser, config.dim_laser_output,
                    config.first_hidden_layer, 
                    config.second_hidden_layer)

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

        for idx in range(len(checkpoint['actor_params'])):
            maddpg.actors[idx].load_state_dict(checkpoint['actor_params'][idx])
        maddpg.actors[-1].load_state_dict(checkpoint['actor_params'][-1])
    else:
        start_epi = config.start_epi

    # tensorboard --logdir=runs
    writer = SummaryWriter()
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

        done = False
        rate = rospy.Rate(10)

        while not done:
            h_in_list = h_out_list
            obs = obs.type(FloatTensor)
            laser = laser.type(FloatTensor)

            # h_out_list : n_agents x num_output(tuple) x (num_layer * num_direction) x batch x hidden
            torch_action, h_out_list = maddpg.select_action(obs, laser, h_in_list)
            action = torch_action.data.cpu()
            obs_, laser_, done, _, _ = env.step(action.numpy().tolist())

            obs_ = torch.from_numpy(obs_).float()
            laser_ = torch.from_numpy(laser_).float()

            obs = obs_
            laser = laser_

if __name__ == '__main__':
    main()