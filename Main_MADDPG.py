#!/usr/bin/env python
# ROS
import rospy

# Env
import os
import numpy as np
import random, collections
import math
import time
import argparse

# RL training
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple

# Parameter
from easydict import EasyDict
import json

# Tensorboard
from torch.utils.tensorboard import SummaryWriter

# Utils
import Utils
from Env import Gazebo_Env
from Train_Utils import soft_update, hard_update, OrnsteinUhlenbeckProcess

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--test', type=bool, default=False)
args = parser.parse_args()

##############################################################################
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
        "episodes_before_train" : 0,
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
        if (args.test == False):
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

            # : (batch_size_non_final) x n_agents x dim_obs
            non_final_next_states = torch.stack([s for s in batch.next_states if s is not None]).type(FloatTensor)
            non_final_next_lasers = torch.stack([l for l in batch.next_laser if l is not None]).type(FloatTensor)

            # for current agent
            whole_state = state_batch.view(self.batch_size, -1)
            whole_laser = laser_batch.view(self.batch_size, -1)
            whole_action = action_batch.view(self.batch_size, -1)
            self.critic_optimizer[agent].zero_grad()
            current_Q = self.critics[agent](whole_state, whole_action, whole_laser)

            # n_agents x batch_size_non_final x next_action
            non_final_next_actions = [self.actors_target[i](non_final_next_states[:, i, :], non_final_next_lasers[:, i, :]) for i in range(self.n_agents)]
            non_final_next_actions = torch.stack(non_final_next_actions)

            # batch_size_non_final x n_agents x next_action
            non_final_next_actions = (non_final_next_actions.transpose(0, 1).contiguous())

            target_Q = torch.zeros(self.batch_size).type(FloatTensor)
            target_Q[non_final_mask] = self.critics_target[agent](
                non_final_next_states.view(-1, self.n_agents * self.n_states),
                non_final_next_actions.view(-1, self.n_agents * self.n_actions),
                non_final_next_lasers.view(-1, self.n_agents * self.n_laser)
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
            action_i = self.actors[agent](state_i, laser_i)
            ac = action_batch.clone()
            ac[:, agent, :] = action_i
            whole_action = ac.view(self.batch_size, -1)

            # check replace true action(from buffer) to each agent's policy from obs --> make whole action from self.actor[agent](state_batch[:, agent, :])
            actor_loss = -self.critics[agent](whole_state, whole_action, whole_laser).mean()
            
            # check performance
            # actor_loss += (action_i ** 2).mean() * 1e-3 # from openai reference code

            actor_loss.backward()
            self.actor_optimizer[agent].step()

            c_loss.append(loss_Q)
            a_loss.append(actor_loss)

        if self.steps_done % 100 == 0 and self.steps_done > 0:
            for i in range(self.n_agents):
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)

        return c_loss, a_loss

    def select_action(self, state_batch, laser_batch):
        # state_batch: n_agents x state_dim
        actions = torch.zeros(
            self.n_agents,
            self.n_actions)
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        for i in range(self.n_agents):
            sb = state_batch[i, :].detach()
            lb = laser_batch[i, :].detach()
            act = self.actors[i](sb.unsqueeze(0), lb.unsqueeze(0)).squeeze()

            act += torch.from_numpy(
                np.random.randn(self.n_actions) * self.var[i]).type(FloatTensor)

            if self.episode_done > self.episodes_before_train and self.var[i] > 0.05:
                self.var[i] *= 0.999998
            act = torch.clamp(act, -1.0, 1.0)

            actions[i, :] = act
        self.steps_done += 1

        return actions
##############################################################################################################
# MADDPG Replay Buffer
Experience = namedtuple('Experience',
                        ('states', 'laser', 'actions', 'next_states', 'next_laser', 'rewards'))

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
        self.FC1 = nn.Linear(obs_dim + 10 + act_dim, hidden_layer)
        self.FC2 = nn.Linear(hidden_layer, second_hidden_layer)
        self.FC3 = nn.Linear(second_hidden_layer, 1)

        # self.PreFC = nn.Linear(las_dim, 10)
        # self.FC1 = nn.Linear(obs_dim + 10, hidden_layer)
        # self.FC2 = nn.Linear(hidden_layer+act_dim, second_hidden_layer)
        # self.FC3 = nn.Linear(second_hidden_layer, 1)

    # obs: batch_size * obs_dim
    def forward(self, obs, acts, laser):
        result = F.relu(self.PreFC(laser))
        combined = torch.cat([result, obs, acts], 1)
        result = F.relu(self.FC1(combined))
        result = F.relu(self.FC2(result))

        # result = F.relu(self.PreFC(laser))
        # combined = torch.cat([result, obs], 1)
        # result = F.relu(self.FC1(combined))
        # combined = torch.cat([result, acts], 1)
        # result = F.relu(self.FC2(combined))

        return self.FC3(result)


class Actor(nn.Module):
    def __init__(self, dim_observation, dim_action, laser_cnt, hidden_layer=128, second_hidden_layer=64):
        super(Actor, self).__init__()
        self.PreFC = nn.Linear(laser_cnt, 10)
        self.FC1 = nn.Linear(dim_observation + 10, hidden_layer)
        self.FC2 = nn.Linear(hidden_layer, second_hidden_layer)
        self.FC3 = nn.Linear(second_hidden_layer, dim_action)

    def forward(self, obs, laser):
        result = F.relu(self.PreFC(laser))
        combined = torch.cat([result, obs], 1)
        result = F.relu(self.FC1(combined))
        result = F.relu(self.FC2(result))
        result = torch.tanh(self.FC3(result)) * 0.5
        return result

##################################################################################

def main():
    env = Gazebo_Env(config.n_agents, config.n_targets, config.dim_laser)
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
            if(Utils.isint(file_list[idx][5:-4])):
                if(max_value < int(file_list[idx][5:-4])):
                    max_index = idx
                    max_value = int(file_list[idx][5:-4])
        last_file = file_list[max_index]

        path = args.model_path + '/' + last_file
        print(path)

        checkpoint = torch.load(path)
        start_epi = checkpoint['n_epi']
        # start_epi = 0
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
    rate = rospy.Rate(20)

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

        past_obs_list = []
        past_laser_list= []
        past_action_list = []
        
        done = False
        rr = np.zeros((config.n_agents,))

        while not done:
            n_step += 1
            obs = obs.type(FloatTensor)
            laser = laser.type(FloatTensor)
            action = maddpg.select_action(obs, laser).data.cpu()
            obs_, laser_, done, reward, _ = env.step(action.numpy().tolist())

            reward = torch.FloatTensor(reward).type(FloatTensor)
            obs_ = torch.from_numpy(obs_).float()
            laser_ = torch.from_numpy(laser_).float()

            if done:
                next_obs = None
                next_laser = None
            else:
                next_obs = obs_
                next_laser = laser_

            total_reward += reward.sum()
            rr += reward.cpu().numpy()

            if (n_step < config.delay_step):
                past_obs_list.append(obs.data)
                past_laser_list.append(laser.data)
                past_action_list.append(action)
            else:
                maddpg.memory.push(past_obs_list.pop(0), past_laser_list.pop(0), past_action_list.pop(0), next_obs, next_laser, reward)
                past_obs_list.append(obs.data)
                past_laser_list.append(laser.data)
                past_action_list.append(action)

            obs = next_obs
            laser = next_laser

            if (args.test == False):
                c_loss, a_loss = maddpg.update_policy()
            rate.sleep()

        score += total_reward
        maddpg.episode_done += 1
        reward_record.append(total_reward)

        writer.add_scalar("Reward function", total_reward, i_episode)
        print("Reward : " + str(total_reward))
        print(" ")

        if i_episode == 0:
            model_path = os.path.join("/home/dwkim/RL_ws/src/rl/src", Utils.Save_path)
            if not os.path.isdir(model_path):
                os.makedirs(model_path)
            
            with open(model_path + '/config.json', 'w') as f:
                json.dump(vars(config), f)

        if i_episode % print_interval == 0 and i_episode != 0:
            avg_score_str = str(int(score/print_interval))
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