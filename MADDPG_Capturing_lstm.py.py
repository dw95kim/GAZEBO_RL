#!/usr/bin/env python
# ROS
import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Float32MultiArray, Int32
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import tf

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

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default=None)
args = parser.parse_args()

##############################################################################
# Env Part
d2r = math.pi/180
r2d = 180/math.pi
eps = 0.00000001

# Helper Function
def isint(value):
    try:
        int(value)
        return True
    except ValueError:
        return False

def satmax(data, max_val):
    if(abs(data) > max_val):
        return (data + eps)/abs(data + eps) * max_val
    else:
        return data

def satmin(data, min_val):
    if(abs(data) < min_val):
        return (data + eps)/abs(data + eps) * min_val
    else:
        return data

def distance_2d_vec(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def distance_3d_vec(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)

def mag_2d_vec(a):
    return math.sqrt(a[0]**2 + a[1]**2)

def mag_3d_vec(a):
    return math.sqrt(a[0]**2 + a[1]**2 + a[2]**2)

def norm_2d_vec(a):
    return np.array(a)/float(mag_2d_vec(a))

def norm_3d_vec(a):
    return np.array(a)/float(mag_3d_vec(a))

def cal_angle_2d_vec(a, b):
    mag_a = mag_2d_vec(a)
    mag_b = mag_2d_vec(b)
    inner_prod = a[0]*b[0] + a[1]*b[1]
    if (mag_a * mag_b == 0):
        cos_value = 0
    else:
        cos_value = inner_prod / (mag_a * mag_b)
    return math.acos(cos_value)

def cal_angle_3d_vec(a, b):
    mag_a = mag_3d_vec(a)
    mag_b = mag_3d_vec(b)
    inner_prod = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
    if (mag_a * mag_b == 0):
        cos_value = 0
    else:
        cos_value = inner_prod / (mag_a * mag_b)
    return math.acos(cos_value)

# xmin, xmax, ymin, ymax are the area of obstacle (forbidden area) 
# xmin value is availble
# xmin-eps is not available
# 
#   1         2         3
#       -------------
#       |(forbidden)|
#   4   |     5     |   6
#       |           |
#       -------------
#   7         8         9
#
def check_area(pos, xmin, xmax, ymin, ymax):
    x = pos[0]
    y = pos[1]
    if (x < xmin):
        if (y < ymin):
            return 7
        elif (y > ymax):
            return 1
        else:
            return 4
    elif (x > xmax):
        if (y < ymin):
            return 9
        elif (y > ymax):
            return 3
        else:
            return 6
    else:
        if (y < ymin):
            return 8
        elif (y > ymax):
            return 2
        else:
            return 5

# 
#   1         2         3
#       -------------
#       |(forbidden)|
#   4   |     5     |   6
#       |           |
#       -------------
#   7         8         9
#
# return array is the next target pos
def avoid_planning(cur_pos, target_pos, xmin, xmax, ymin, ymax):
    cur_area = check_area(cur_pos, xmin, xmax, ymin, ymax)
    tar_area = check_area(target_pos, xmin, xmax, ymin, ymax)

    if (cur_area == 1):
        if (tar_area == 8):
            return [xmin, ymin]
        elif (tar_area == 6):
            return [xmax, ymax]
        elif (tar_area == 9):
            temp1 = distance_2d_vec(cur_pos, [xmax, ymax]) + distance_2d_vec([xmax, ymax], tar_pos)
            temp2 = distance_2d_vec(cur_pos, [xmin, ymin]) + distance_2d_vec([xmin, ymin], tar_pos)
            if (temp1 < temp2):
                return [xmax, ymax]
            else:
                return [xmin, ymin]
        else:
            return target_pos
    if (cur_area == 3):
        if (tar_area == 8):
            return [xmax, ymin]
        elif (tar_area == 4):
            return [xmin, ymax]
        elif (tar_area == 7):
            temp1 = distance_2d_vec(cur_pos, [xmax, ymin]) + distance_2d_vec([xmax, ymin], tar_pos)
            temp2 = distance_2d_vec(cur_pos, [xmin, ymax]) + distance_2d_vec([xmin, ymax], tar_pos)
            if (temp1 < temp2):
                return [xmax, ymin]
            else:
                return [xmin, ymax]
        else:
            return target_pos
    if (cur_area == 7):
        if (tar_area == 2):
            return [xmin, ymax]
        elif (tar_area == 6):
            return [xmax, ymin]
        elif (tar_area == 3):
            temp1 = distance_2d_vec(cur_pos, [xmin, ymax]) + distance_2d_vec([xmin, ymax], tar_pos)
            temp2 = distance_2d_vec(cur_pos, [xmax, ymin]) + distance_2d_vec([xmax, ymin], tar_pos)
            if (temp1 < temp2):
                return [xmin, ymax]
            else:
                return [xmax, ymin]
        else:
            return target_pos
    if (cur_area == 9):
        if (tar_area == 2):
            return [xmax, ymax]
        elif (tar_area == 4):
            return [xmin, ymin]
        elif (tar_area == 1):
            temp1 = distance_2d_vec(cur_pos, [xmax, ymax]) + distance_2d_vec([xmax, ymax], tar_pos)
            temp2 = distance_2d_vec(cur_pos, [xmin, ymin]) + distance_2d_vec([xmin, ymin], tar_pos)
            if (temp1 < temp2):
                return [xmax, ymax]
            else:
                return [xmin, ymin]
        else:
            return target_pos
    elif (cur_area == 2):
        if (tar_area == 8):
            if (cur_pos[0] + target_pos[0] < xmin + xmax):
                return [xmin, ymax]
            else:
                return [xmax, ymax]
        elif (tar_area == 4 or tar_area == 7):
            return [xmin, ymax]
        elif (tar_area == 6 or tar_area == 9):
            return [xmax, ymax]
        else:
            return target_pos
    elif (cur_area == 8):
        if (tar_area == 2):
            if (cur_pos[0] + target_pos[0] < xmin + xmax):
                return [xmin, ymin]
            else:
                return [xmax, ymin]
        elif (tar_area == 4 or tar_area == 1):
            return [xmin, ymin]
        elif (tar_area == 6 or tar_area == 3):
            return [xmax, ymin]
        else:
            return target_pos
    elif (cur_area == 4):
        if (tar_area == 6):
            if (cur_pos[1] + target_pos[1] < ymin + ymax):
                return [xmin, ymin]
            else:
                return [xmin, ymax]
        elif (tar_area == 2 or tar_area == 3):
            return [xmin, ymax]
        elif (tar_area == 8 or tar_area == 9):
            return [xmin, ymin]
        else:
            return target_pos
    else: # 6
        if (tar_area == 4):
            if (cur_pos[1] + target_pos[1] < ymin + ymax):
                return [xmax, ymin]
            else:
                return [xmax, ymax]
        elif (tar_area == 1 or tar_area == 2):
            return [xmax, ymax]
        elif (tar_area == 7 or tar_area == 8):
            return [xmax, ymin]
        else:
            return target_pos

def clip(data, min_val, max_val):
    return min(max_val, max(min_val, data))

def QuaternionToDCM(q):
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]

    R = [
        [1 - 2*y**2 - 2*z**2,       2*x*y + 2*w*z,          2*x*z - 2*w*y],
        [2*x*y - 2*w*z,             1-2*x**2-2*z**2,        2*y*z+2*w*x],
        [2*x*z + 2*w*y,             2*y*z-2*w*x,            1-2*x**2-2*y**2]
    ]

    return np.array(R)

##############################################################################
# Save Folder
year = datetime.datetime.today().year

month = datetime.datetime.today().month
if(len(str(month)) == 1):   
    month = "0" + str(month)

day = datetime.datetime.today().day
if(len(str(day)) == 1):
    day = "0" + str(day)

hour = datetime.datetime.today().hour
if(len(str(hour)) == 1):
    hour = "0" + str(hour)

minute = datetime.datetime.today().minute
if(len(str(minute)) == 1):
    minute = "0" + str(minute)

Save_path = "model/" + str(year) + "_" + str(month) + "_" + str(day) + "_" + str(hour) + "_" + str(minute) + "_save_model"
Start_time = str(year) + "_" + str(month) + str(day) + "_" + str(hour) + str(minute)

##################################################################################
# RL Part
##################################################################################
#Hyperparameters

if (args.model_path == None):
    config = EasyDict({
        "start_epi"             : 0,
        "n_agents"              : 2,
        "n_targets"             : 1,
        "dim_obs"               : 25,
        "dim_act"               : 3,
        "batch_size"            : 1024,
        "capacity"              : 1000000,
        "lr_ac"                 : 0.0001,
        "lr_cr"                 : 0.001,
        "gamma"                 : 0.95,
        "first_hidden_layer"    : 128,
        "second_hidden_layer"   : 64,
        "tau"                   : 0.01,  # for target network soft update
        "delay_step"            : 10,
        "n_episode"             : 2000,
        "episodes_before_train" : 5,
        "dim_laser"             : 100
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
        "lr_ac"                 : json_data["lr_ac"],
        "lr_cr"                 : json_data["lr_cr"],
        "gamma"                 : json_data["gamma"],
        "first_hidden_layer"    : json_data["first_hidden_layer"],
        "second_hidden_layer"   : json_data["second_hidden_layer"],
        "tau"                   : json_data["tau"],  # for target network soft update
        "delay_step"            : json_data["delay_step"],
        "n_episode"             : json_data["n_episode"],
        "episodes_before_train" : json_data["episodes_before_train"],
        "dim_laser"             : json_data["dim_laser"]
    })


##############################################################################################################
# MADDPG Random Process
class RandomProcess:
    def reset_states(self):
        pass


class AnnealedGaussianProcess(RandomProcess):
    def __init__(self, mu, sigma, sigma_min, n_steps_annealing):
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma


class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):
    def __init__(self, theta, mu=0., sigma=0.2,
                 dt=1e-2, x0=None, size=1,
                 sigma_min=None, n_steps_annealing=1000):
        super(OrnsteinUhlenbeckProcess,
              self).__init__(mu=mu,
                             sigma=sigma,
                             sigma_min=sigma_min,
                             n_steps_annealing=n_steps_annealing)
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self):
        x = self.x_prev + \
            self.theta * (self.mu -
                          self.x_prev) * self.dt + (
                              self.current_sigma * np.sqrt(self.dt) *
                              np.random.normal(size=self.size)
                              )
        self.x_prev = x
        self.n_steps += 1
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)

##############################################################################################################
# MADDPG Train Part
def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(source_param.data)


class MADDPG:
    def __init__(self, n_agents, dim_obs, dim_act, dim_laser, batch_size, first_hidden_layer, second_hidden_layer,
                 capacity, episodes_before_train, gamma, tau, lr_cr, lr_ac):

        self.actors = [Actor(dim_obs, dim_act, dim_laser, first_hidden_layer, second_hidden_layer) for i in range(n_agents)]
        self.critics = [Critic(n_agents, dim_obs, dim_act, dim_laser, first_hidden_layer, second_hidden_layer) for i in range(n_agents)]

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

        self.var = [1.0 for i in range(n_agents)]
        cnt = 0
        while (self.var[0] > 0.05 and cnt < config.start_epi - episodes_before_train):
            for i in range(len(self.var)):
                self.var[i] *= 0.999998
            cnt += 1
        
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
            # h_in : batch x agent x (num_layer * direction) x hidden parameter
            h_in = torch.stack(batch.h_in).type(FloatTensor)
            h_out = torch.stack(batch.h_out).type(FloatTensor)

            # : (batch_size_non_final) x n_agents x dim_obs
            non_final_next_states = torch.stack([s for s in batch.next_states if s is not None]).type(FloatTensor)
            non_final_next_lasers = torch.stack([l for l in batch.next_laser if l is not None]).type(FloatTensor)

            # : (batch_size_non_final) x agent x (num_layer * direction) x hidden paramter
            non_final_next_h_out = torch.stack([h for h in batch.h_out if h is not None]).type(FloatTensor)
            non_final_next_h_out = non_final_next_h_out.transpose(0, 2) # : (num_layer * direction) x agent x batch_size_non_final x hidden parameter
            non_final_next_h_out = non_final_next_h_out.transpose(1, 2) # : (num_layer * direction) x batch_size_non_final x agent x hidden parameter

            # for current agent
            whole_state = state_batch.view(self.batch_size, -1)
            whole_laser = laser_batch.view(self.batch_size, -1)
            whole_action = action_batch.view(self.batch_size, -1)
            
            h_in = h_in.transpose(0, 2) # h_in : (num_layer * direction) x agent x batch x hidden parameter
            h_in = h_in.transpose(1, 2) # h_in : (num_layer * direction) x batch x agent x hidden parameter

            # whole_h_in : (num_layer * direction) x batch_size x hidden
            whole_h_in = h_in.view(1, self.batch_size, -1)
            self.critic_optimizer[agent].zero_grad()
            current_Q = self.critics[agent](whole_state, whole_actions, whole_laser, whole_h_in)

            # n_agents x batch_size_non_final x next_action
            non_final_next_actions = [self.actors_target[i](non_final_next_states[:, i, :], non_final_next_lasers[:, i, :]) for i in range(self.n_agents)]
            non_final_next_actions = torch.stack(non_final_next_actions)

            # batch_size_non_final x n_agents x next_action
            non_final_next_actions = (non_final_next_actions.transpose(0, 1).contiguous())

            target_Q = torch.zeros(self.batch_size).type(FloatTensor)
            target_Q[non_final_mask] = self.critics_target[agent](
                non_final_next_states.view(-1, self.n_agents * self.n_states),
                non_final_next_actions.view(-1, self.n_agents * self.n_actions),
                non_final_next_lasers.view(-1, self.n_agents * self.n_laser),
                non_final_next_h_out.view(1, self.batch_size, -1)
            ).squeeze()

            # TD Target = r + gamma * target_Q
            # TD Target shape : batch_size x 1 (agent)
            target_Q = (target_Q.unsqueeze(1) * self.GAMMA) + (reward_batch[:, agent].unsqueeze(1))

            # Update Critic Network
            loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
            loss_Q.backward()
            self.critic_optimizer[agent].step()

            # Update Actor Network
            self.actor_optimizer[agent].zero_grad()
            state_i = state_batch[:, agent, :]
            laser_i = laser_batch[:, agent, :]

            # h_in : (num_layer * direction) x batch x agent x hidden parameter
            h_in_i = h_in[:, :, agent, :]
            action_i, h_out_i = self.actors[agent](state_i, laser_i, h_in_i)
            ac = action_batch.clone()
            ac[:, agent, :] = action_i
            whole_action = ac.view(self.batch_size, -1)

            # check replace true action(from buffer) to each agent's policy from obs --> make whole action from self.actor[agent](state_batch[:, agent, :])
            actor_loss = -self.critics[agent](whole_state, whole_action, whole_laser, whole_h_in).mean()
            
            # check performance
            actor_loss += (action_i ** 2).mean() * 1e-3 # from openai reference code

            actor_loss.backward()
            self.actor_optimizer[agent].step()

            c_loss.append(loss_Q)
            a_loss.append(actor_loss)

        if self.steps_done % 100 == 0 and self.steps_done > 0:
            for i in range(self.n_agents):
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)

        return c_loss, a_loss

    def select_action(self, state_batch, laser_batch, h_in_batch):
        # state_batch: n_agents x state_dim
        actions = torch.zeros(
            self.n_agents,
            self.n_actions)
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        # agent x (num_layer x direction) x batch x hidden
        next_lstm_hidden = torch.zeros((self.n_agents, 1, 1, config.second_hidden_layer))

        for i in range(self.n_agents):

            #  state_batch : nagents x state count
            sb = state_batch[i, :].detach()
            lb = laser_batch[i, :].detach()

            # h_in : agent x (num_layer * direction) x hidden parameter
            h_in = h_in_batch[i, :, :].detach()

            # make batch / because nn only take ONLY sample 'batchs' NOT simple input 
            # need to make (num_layer * direction) x batch x hidden_parameter
            act, lstm_hidden = self.actors[i](sb.unsqueeze(0), lb.unsqueeze(0), h_in.unsqueeze(1)).squeeze()

            act += torch.from_numpy(
                np.random.randn(self.n_actions) * self.var[i]).type(FloatTensor)

            if self.episode_done > self.episodes_before_train and self.var[i] > 0.05:
                self.var[i] *= 0.999998
            act = torch.clamp(act, -1.0, 1.0)

            actions[i, :] = act

            next_lstm_hidden[i] = lstm_hidden
        self.steps_done += 1

        return actions, next_lstm_hidden
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
    def __init__(self, n_agent, dim_observation, dim_action, laser_cnt, hidden_layer=128, second_hidden_layer=64):
        super(Critic, self).__init__()
        self.n_agent = n_agent
        self.dim_observation = dim_observation
        self.dim_action = dim_action
        obs_dim = dim_observation * n_agent
        act_dim = self.dim_action * n_agent
        las_dim = laser_cnt * n_agent

        self.PreFC = nn.Linear(las_dim, 10)
        self.FC1 = nn.Linear(obs_dim + 10, hidden_layer)
        # self.FC2 = nn.Linear(hidden_layer + act_dim, second_hidden_layer)
        self.LSTM = nn.LSTM(hidden_layer + act_dim, second_hidden_layer)
        self.FC3 = nn.Linear(second_hidden_layer, 1)

    # obs: batch_size * obs_dim
    def forward(self, obs, acts, laser, hidden):
        result = F.relu(self.PreFC(laser))
        combined = torch.cat([result, obs], 1)
        result = F.relu(self.FC1(combined))
        combined = torch.cat([result, acts], 1)
        # result = F.relu(self.FC2(combined))
        result, lstm_hidden = self.LSTM(combined, hidden)
        return self.FC3(result)


class Actor(nn.Module):
    def __init__(self, dim_observation, dim_action, laser_cnt, hidden_layer=128, second_hidden_layer=64):
        super(Actor, self).__init__()
        self.PreFC = nn.Linear(laser_cnt, 10)
        self.FC1 = nn.Linear(dim_observation + 10, hidden_layer)
        # self.FC2 = nn.Linear(hidden_layer, second_hidden_layer)
        self.LSTM = nn.LSTM(hidden_layer, second_hidden_layer)
        self.FC3 = nn.Linear(second_hidden_layer, dim_action)

    def forward(self, obs, laser, hidden):
        result = F.relu(self.PreFC(laser))
        combined = torch.cat([result, obs], 1)
        result = F.relu(self.FC1(combined))
        # result = F.relu(self.FC2(result))
        result, lstm_hidden = self.LSTM(result, hidden)
        result = torch.tanh(self.FC3(result)) * 0.5
        return result, lstm_hidden

##################################################################################

class Gazebo_Env():
    def __init__(self):
        rospy.init_node('rl_node')
    
        # n_agents : 2, n_target : 1
        # [0, 1, 2] --> [agent, agent, target] --> start_target index = 2 = n_agents
        self.start_target_index = config.n_agents
        self.total_drone = config.n_agents + config.n_targets

        # i --> Agent i th pose / velocity / angle
        self.pos_list = np.zeros((self.total_drone, 3))
        self.vel_list = np.zeros((self.total_drone, 3))
        self.angle_list = np.zeros((self.total_drone, 3))
        
        # i,j --> Agent i pos/vel from Agent j frame
        self.pos_err_jtoi_from_j_frame = np.zeros((self.total_drone, self.total_drone, 3))
        self.vel_err_jtoi_from_j_frame = np.zeros((self.total_drone, self.total_drone, 3))

        # [x_vel, y_vel, z_vel, throttle]
        self.vel_cmd_list = np.zeros((config.n_agents, 4))

        # Subscribe Node
        for i in range(self.total_drone):
            rospy.Subscriber('/UAV_' + str(i) + '/mavros/local_position/pose', PoseStamped, self.callback_pos, (i))
            rospy.Subscriber('/UAV_' + str(i) + '/mavros/local_position/velocity_local', TwistStamped, self.callback_vel, (i))
            rospy.Subscriber('/UAV_' + str(i) + '/scan', LaserScan, self.callback_laser, (i))

        # Publish Node : mission / goal x, y, z / goal yaw / max vel_x_y, vel_z
        self.pub_list = []
        for i in range(self.total_drone):
            self.pub_list.append(rospy.Publisher('/UAV_' + str(i) + '/GoalAction', Float32MultiArray, queue_size=1))

        self.check_input_pub = rospy.Publisher("/check/input", Float32MultiArray, queue_size=1)

        #############################
        # For Env and Training
        #############################
        self.goal_list = []
        for i in range(config.n_agents + config.n_targets):
            self.goal_list.append(Float32MultiArray())

        self.set_first_start_time = True

        # self.check_input = input_type()
        self.check_input = Float32MultiArray()

        self.state = []
        self.start_time = time.time()
        self.prev_callback_time = 0
        self.prev_step_time = 0

        # check delay step
        self.file = None
        self.epi = 0
        self.rl_start = False
        self.first_step = True

        # Done Part
        self.capture_done = 0
        self.collision_done = 0
        self.out_field_done = 0

        # Debug Part
        self.map_size = [20, 20]
        self.inital_position = [np.array([3, 3, 10]), np.array([15, 15, 10]), np.array([10, 2.5, 10])]
        self.offset_list = [np.array([3, 3, 0]), np.array([15, 15, 0]), np.array([10, 2.5, 0])]
        self.start_angle = [0.0, 3.141592, -1.57079]
        self.time_constrain = 100
        self.min_distance = 1.5
        self.Off_field_constrain = 100
        self.agent_fov = 120 # drgree
        self.target_manuver = 1 # 1 : Circle , 2 : Avoiding Alg
        self.check_callback_Hz = False 
        self.check_step_Hz = False
        self.check_print_state = False
        self.write_command_file = False
        self.laser_list = np.zeros((config.n_agents, config.dim_laser))
        self.reward_type = 1 # 1 : Sparse, 2 : Dense

    def callback_laser(self, data, agents):
        sample_index = int(1081 / config.dim_laser)
        temp_laser = []
        for i in range(config.dim_laser):
            if (data.ranges[i * sample_index] == np.inf):
                value = 0
            else:
                value = (20 - data.ranges[i * sample_index]) / 20.0
            temp_laser.append(value)
        self.laser_list[agents] = np.array(temp_laser)

    def callback_vel(self, data, agents):
        self.vel_list[agents] = np.array([data.twist.linear.x, data.twist.linear.y, data.twist.linear.z])

    def callback_pos(self, data, agents):
        q = [
            data.pose.orientation.x,
            data.pose.orientation.y,
            data.pose.orientation.z,
            data.pose.orientation.w
        ]

        rot_matrix = QuaternionToDCM(q)

        self.pos_list[agents] = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z]) + self.offset_list[agents]

        for i in range(self.total_drone):
            self.pos_err_jtoi_from_j_frame[i][agents] = np.matmul(rot_matrix, self.pos_list[i] - self.pos_list[agents])

        self.angle_list[agents] = np.array(list(tf.transformations.euler_from_quaternion(q))) / 2.0

        for i in range(self.total_drone):
            self.vel_err_jtoi_from_j_frame[i][agents] = np.matmul(rot_matrix, self.vel_list[i])
    
    def make_state(self):
        # target pos, target vel, target yaw, own pos, own vel, own yaw, own vel cmd, friends pos, friends vel, friends yaw, 
        # 3 3 1 / 3 3 1 4 / 3 3 1 ==> 25
        state_list = [0 for i in range(config.n_agents)]
        for i in range(config.n_agents):
            state_list[i] = self.pos_list[self.start_target_index].tolist() + self.vel_list[self.start_target_index].tolist() + [self.angle_list[self.start_target_index][2]]
            state_list[i] = state_list[i] + self.pos_list[i].tolist() + self.vel_list[i].tolist() + [self.angle_list[i][2]] + self.vel_cmd_list[i].tolist()
            for j in range(config.n_agents):
                if j != i:
                    state_list[i] = state_list[i] + self.pos_list[j].tolist() + self.vel_list[j].tolist() + [self.angle_list[j][2]]
            state_list[i] = np.array(state_list[i])

        # Mask Part --> Convert Zero Value
        isdetect = 0 # 0 : can not detect target drone / 1 : can detect target drone
        for i in range(config.n_agents):
            if (self.angle_list[i][2] == 0):
                self.angle_list[i][2] += 1e-5

            heading_vector = [math.cos(self.angle_list[i][2]), math.sin(self.angle_list[i][2])]
            vector_itoj = self.pos_list[self.start_target_index][:2] - self.pos_list[i][:2]

            if (cal_angle_2d_vec(heading_vector, vector_itoj) < self.agent_fov / 2 * d2r):
                # Check Obstacle
                y1 = (self.pos_list[i][1] - self.pos_list[self.start_target_index][1])/(self.pos_list[i][0] - self.pos_list[self.start_target_index][0]) * (10 - self.pos_list[i][0]) + self.pos_list[i][1]
                y2 = (self.pos_list[i][1] - self.pos_list[self.start_target_index][1])/(self.pos_list[i][0] - self.pos_list[self.start_target_index][0]) * (15 - self.pos_list[i][0]) + self.pos_list[i][1]
                x1 = (self.pos_list[i][0] - self.pos_list[self.start_target_index][0])/(self.pos_list[i][1] - self.pos_list[self.start_target_index][1]) * (10 - self.pos_list[i][1]) + self.pos_list[i][0]
                if not (7.5 < y1 < 12.5 or 7.5 < y2 < 12.5 or 7.5 < x1 < 17.5):
                    isdetect = 1

        if (isdetect == 0):
            for i in range(config.n_agents):
                state_list[i][:7] = [0, ]

        self.state = np.row_stack(tuple([i for i in state_list]))
        self.check_input.data = self.state

    def reset(self):
        self.epi += 1
        self.set_first_start_time = True

        self.capture_done = 0
        self.collision_done = 0
        self.out_field_done = 0

        past_initial_position_list = []
        for i in range(self.total_drone):
            past_initial_position_list.append([self.pos_list[i][0], self.pos_list[i][1], 10 * (i+2)])

        zero_initial_position_list = []
        for i in range(self.total_drone):
            zero_initial_position_list.append([0.0, 0.0, 10 * (i+2)])

        new_initial_position_list = []
        for i in range(self.total_drone):
            new_initial_position_list.append([self.inital_position[i][0], self.inital_position[i][1], 10 * (i+2)])

        zero_same_alt_list = []
        for i in range(self.total_drone):
            zero_same_alt_list.append([0.0, 0.0, 10.0])
            
        # Seperate Altitude (tracking0 : 20, tracing1 : 30, target : 40)
        while (abs(np.amax(self.pos_list - past_initial_position_list, axis=0)[2]) > 1.0):
            for i in range(self.total_drone):
                self.goal_list[i].data = [3.0] + past_initial_position_list[i] + [self.start_angle[i]] + [0.0, 1.0]
                self.pub_list[i].publish(self.goal_list[i])
        print("PASS Seperate Alt")

        # Moving Specific position
        while (True):
            for i in range(self.total_drone):
                self.goal_list[i].data = [3.0] + zero_initial_position_list[i] + [self.start_angle[i]] + [1.0, 1.0]
                self.pub_list[i].publish(self.goal_list[i])
        
            distance_list = []
            for i in range(self.total_drone):
                distance_list.append(distance_3d_vec(self.pos_list[i], new_initial_position_list[i]))
            if (max(distance_list) < 2):
                break
        print("PASS Reset Position")

        # Fit Same Altitude
        while (abs(np.amax(self.pos_list - self.inital_position, axis=0)[2]) > 1.0):
            for i in range(self.total_drone):
                self.goal_list[i].data = [3.0] + zero_same_alt_list[i] + [self.start_angle[i]] + [0.0, 1.0]
                self.pub_list[i].publish(self.goal_list[i])
        print("PASS SAME Alt")

        self.make_state()
        return self.state, self.laser_list

    # action torch type
    # [[agent1_x_vel, agent1_y_vel, agent1_r_vel],
    # [agent2_x_vel, agent2_y_vel, agent2_r_vel]]
    def step(self, action):
        if (self.set_first_start_time == True):
            self.start_time = time.time()
            self.set_first_start_time = False

        for i in range(config.n_agents):
            self.vel_cmd_list[i] = [action[i][0], action[i][1], satmax(1.5*(self.inital_position[i][2] - self.pos_list[i][2]), 1) + 0.3*(0.0 - self.vel_list[i][2]), action[i][2]]
            self.goal_list[i].data = [7.0] + self.vel_cmd_list[i].tolist() + [0.0, 0.0]

        # Circle 
        if (self.target_manuver == 1):
            center_point = np.array([10, 10])
            center_to_target = self.pos_list[self.start_target_index][:2] - center_point
            vel_vec = [-center_to_target[1], +center_to_target[0]]
            normal_vel_vec = norm_2d_vec(vel_vec)
            final_heading = math.atan(normal_vel_vec[1]/normal_vel_vec[0])

            # make z command
            z_vel = satmax(1.5*(self.inital_position[self.start_target_index][2] - self.pos_list[self.start_target_index][2]), 1.0) + 0.3*(0.0 - self.vel_list[self.start_target_index][2])

            self.goal_list[self.start_target_index].data = [9.0] + normal_vel_vec.tolist() + [z_vel] + [final_heading] + [0.0, 0.0]
        elif (self.target_manuver == 2):
            min_point = [-1, -1]
            max_distance = -1
            # check all point in egde and corner
            for i in range(self.map_size[0]):
                min_distance_between_target = min(distance_2d_vec(self.pos_list[0][:2], [i, 0]), distance_2d_vec(self.pos_list[1][:2], [i, 0]))
                if (max_distance < min_distance_between_target):
                    max_distance = min_distance_between_target
                    min_point = [i, 0]
            for i in range(self.map_size[1]):
                min_distance_between_target = min(distance_2d_vec(self.pos_list[0][:2], [0, i]), distance_2d_vec(self.pos_list[1][:2], [0, i]))
                if (max_distance < min_distance_between_target):
                    max_distance = min_distance_between_target
                    min_point = [0, i]
            for i in range(self.map_size[0]):
                min_distance_between_target = min(distance_2d_vec(self.pos_list[0][:2], [i, self.map_size[1]]), distance_2d_vec(self.pos_list[1][:2], [i, self.map_size[1]]))
                if (max_distance < min_distance_between_target):
                    max_distance = min_distance_between_target
                    min_point = [i, self.map_size[1]]
            for i in range(self.map_size[1]):
                min_distance_between_target = min(distance_2d_vec(self.pos_list[0][:2], [self.map_size[0], i]), distance_2d_vec(self.pos_list[1][:2], [self.map_size[0], i]))
                if (max_distance < min_distance_between_target):
                    max_distance = min_distance_between_target
                    min_point = [self.map_size[0], i]

            # cur_pos, target_pos, xmin, xmax, ymin, ymax
            next_target_pos = avoid_planning(self.pos_list[self.start_target_index][:2], min_point, 7, 18, 7, 13)
            heading_vector = (np.array(next_target_pos) - self.pos_list[self.start_target_index][:2])
            heading_angle = math.atan(heading_vector[1]/heading_vector[0])

            next_target_pos = (np.array(next_target_pos) - self.offset_list[self.start_target_index][:2]).tolist()
            # mission number, target (x, y), [z, heading], [max_vel_hor, max_vel_ver]
            self.goal_list[self.start_target_index].data = [8.0] + next_target_pos + [10, heading_angle] + [1.0, 1.0]

            # vel_vec = np.array(min_point) - self.pos_list[self.start_target_index][:2]
            # normal_vel_vec = norm_2d_vec(vel_vec)
            # final_heading = math.atan(normal_vel_vec[1]/normal_vel_vec[0])

            # # make z command
            # z_vel = satmax(1.5*(self.inital_position[self.start_target_index][2] - self.pos_list[self.start_target_index][2]), 1.0) + 0.3*(0.0 - self.vel_list[self.start_target_index][2])

            # self.goal_list[self.start_target_index].data = [8.0] + normal_vel_vec.tolist() + [z_vel] + [final_heading] + [0.0, 0.0]

        # Publish part
        for i in range(self.total_drone):
            self.pub_list[i].publish(self.goal_list[i])
        self.check_input_pub.publish(self.check_input)

        # make state part
        self.make_state()

        return self.state, self.laser_list, self.is_gameover(), self.get_reward(), None
        
    def is_gameover(self):
        if (time.time() - self.start_time > self.time_constrain):
            print("Time Constrain")
            if (self.write_command_file == True):
                self.file.close()
                self.file = None
            return True

        distance_target_list = []
        for i in range(self.start_target_index, self.total_drone):
            for j in range(config.n_agents):
                distance_target_list.append(mag_3d_vec(self.pos_err_jtoi_from_j_frame[i][j]))
        if (min(distance_target_list) < self.min_distance):
            self.capture_done = 1
            print("Capture Success")
            if (self.write_command_file == True):
                self.file.close()
                self.file = None
            return True

        distance_drone_list = []
        for i in range(self.start_target_index):
            for j in range(self.start_target_index):
                if (i != j):
                    distance_drone_list.append(mag_3d_vec(self.pos_err_jtoi_from_j_frame[i][j]))
        if (min(distance_drone_list) < self.min_distance):
            self.collision_done = 1
            print("Collision Warning with Tracking Drone")
            if (self.write_command_file == True):
                self.file.close()
                self.file = None
            return True

        for i in range(self.start_target_index):
            min_cnt = 0
            for j in range(config.dim_laser):
                if (self.laser_list[i][j] > 0.93):
                    min_cnt += 1
            if (min_cnt > 10):
                self.collision_done = 1
                print("Collision Warning with Obstacle by laser")
                if (self.write_command_file == True):
                    self.file.close()
                    self.file = None
                return True

            if(distance_2d_vec(self.pos_list[i][:2], np.array([10, 10])) < 3.5 or distance_2d_vec(self.pos_list[i][:2], np.array([15, 10])) < 3.5):
                self.collision_done = 1
                print("Collision Warning with Obstacle by distance")
                if (self.write_command_file == True):
                    self.file.close()
                    self.file = None
                return True

        for i in range(self.total_drone):
            if (self.pos_list[i][0] < -5 or self.pos_list[i][0] > 25 or self.pos_list[i][0] < -5 or self.pos_list[i][1] > 25):
                self.out_field_done = 1
                print("Off field")
                if (self.write_command_file == True):
                    self.file.close()
                    self.file = None
                return True

    def get_reward(self):
        # check whether drone close to target
        target_distance_list = []
        for i in range(config.n_agents):
            target_distance_list.append(distance_3d_vec(self.pos_list[self.start_target_index], self.pos_list[i]))

        if (self.reward_type == 1): # Sparse Reward
            reward = np.zeros(config.n_agents)
            for i in range(config.n_agents):
                if (self.capture_done == 1):
                    reward[i] += 1000
                if (self.collision_done == 1):
                    reward[i] += -1000
                if (self.out_field_done == 1):
                    reward[i] += -1000
                reward[i] -= 1
        elif (self.reward_type == 2): # Dense Reward
            distance_reward_list = np.zeros(config.n_agents)
            penalty_reward_list = np.zeros(config.n_agents)

            for target_distance in target_distance_list:
                # Distance
                if (target_distance < 2):
                    distance_reward = 1000 - (time.time() - self.start_time) * 10
                elif (target_distance < 20):
                    distance_reward = 1 - target_distance / 20.0
                else:
                    distance_reward = 0
                distance_reward_list[i] = distance_reward

            for pos in self.pos_list[:self.start_target_index]:
                # Collision and Defense
                if (mag_3d_vec(pos) > self.Off_field_constrain):
                    penalty_reward = -100
                else:
                    penalty_reward = 0
                penalty_reward_list[i] = penalty_reward

            reward = 1.0 * distance_reward_list + penalty_reward_list
        return reward

##################################################################################
# MADDPG Main

def main():
    env = Gazebo_Env()
    reward_record = []

    print_interval = 10
    score = 0

    maddpg = MADDPG(config.n_agents, config.dim_obs, config.dim_act, config.dim_laser, config.batch_size, config.first_hidden_layer, config.second_hidden_layer, 
                    config.capacity, config.episodes_before_train, config.gamma, config.tau, config.lr_cr, config.lr_ac)

    if (args.model_path != None):
        file_list = os.listdir(args.model_path)

        max_index = 0
        max_value = 0
        for idx in range(len(file_list)):
            if(isint(file_list[idx][5:-4])):
                if(max_value < int(file_list[idx][5:-4])):
                    max_index = idx
                    max_value = int(file_list[idx][5:-4])
        last_file = file_list[max_index]

        path = args.model_path + '/' + last_file
        print(path)

        checkpoint = torch.load(path)
        for a, aopt, params, opt in zip(maddpg.actors, maddpg.actor_optimizer, checkpoint['actor_params'], checkpoint['actor_optim']):
            a.load_state_dict(params)
            aopt.load_state_dict(opt)
        for a, aopt, params, opt in zip(maddpg.critics, maddpg.critic_optimizer, checkpoint['critic_params'], checkpoint['critic_optim']):
            a.load_state_dict(params)
            aopt.load_state_dict(opt)

    # tensorboard --logdir=runs
    writer = SummaryWriter()
    rate = rospy.Rate(20)

    sleep(3)
    print("Start Training")

    FloatTensor = torch.cuda.FloatTensor if maddpg.use_cuda else torch.FloatTensor
    for i_episode in range(config.start_epi, config.n_episode):
        obs, laser = env.reset()
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        if isinstance(laser, np.ndarray):
            laser = torch.from_numpy(laser).float()

        # h_in : agent x (num_layer * direction) x hidden parameter
        h_out = torch.zeros((config.n_agents, 1, config.second_hidden_layer)).type(FloatTensor)

        total_reward = 0.0
        n_step = 0

        past_obs_list = []
        past_laser_list= []
        past_action_list = []
        past_h_in_list = []
        past_h_out_list = []
        
        done = False
        rr = np.zeros((config.n_agents,))

        while not done:
            n_step += 1
            h_in = h_out
            obs = obs.type(FloatTensor)
            laser = laser.type(FloatTensor)

            # h_out : agent x (num_layer * direction) x batch x hidden_parameter
            action, h_out = maddpg.select_action(obs, laser, h_in).data.cpu()
            obs_, laser_, done, reward, _ = env.step(action.numpy().tolist())

            reward = torch.FloatTensor(reward).type(FloatTensor)
            obs_ = torch.from_numpy(obs_).float()
            laser_ = torch.from_numpy(laser_).float()

            # squeeze batch dimension
            h_out = h_out.squeeze(2)

            if done:
                next_obs = None
                next_laser = None
                h_out = None
            else:
                next_obs = obs_
                next_laser = laser_

            total_reward += reward.sum()
            rr += reward.cpu().numpy()

            if (n_step < config.delay_step):
                past_obs_list.append(obs.data)
                past_laser_list.append(laser.data)
                past_action_list.append(action)
                past_h_in_list.append(h_in)
                past_h_out_list.append(h_out)
            else:
                maddpg.memory.push(past_obs_list.pop(0), past_laser_list.pop(0), past_action_list.pop(0), next_obs, next_laser, reward, past_h_in_list.pop(0), past_h_out_list.pop(0))
                past_obs_list.append(obs.data)
                past_laser_list.append(laser.data)
                past_action_list.append(action)
                past_h_in_list.append(h_in)
                past_h_out_list.append(h_out)

            obs = next_obs
            laser = next_laser

            c_loss, a_loss = maddpg.update_policy()
            rate.sleep()

        score += total_reward
        maddpg.episode_done += 1
        reward_record.append(total_reward)

        writer.add_scalar("Reward function", total_reward, i_episode)
        print("Reward : " + str(total_reward))
        print(" ")

        if i_episode == 0:
            model_path = os.path.join("/home/dwkim/RL_ws/src/rl/src", Save_path)
            if not os.path.isdir(model_path):
                os.makedirs(model_path)
            
            with open(model_path + '/config.json', 'w') as f:
                json.dump(vars(config), f)

        if i_episode % print_interval == 0 and i_episode != 0:
            avg_score_str = str(int(score/print_interval))
            ckpt_path = os.path.join(model_path, 'ckpt_'+ avg_score_str + '.pth')

            torch.save(
                {
                'actor_params' : [a.state_dict() for a in maddpg.actors],
                'critic_params' : [a.state_dict() for a in maddpg.critics],
                'actor_optim' : [a.state_dict() for a in maddpg.actor_optimizer],
                'critic_optim' : [a.state_dict() for a in maddpg.critic_optimizer],
                }, ckpt_path)

            print("# of episode :{}, avg score : {:.1f}".format(i_episode, score/print_interval))
            score = 0.0
        
if __name__ == '__main__':
    main()