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
from Env_3action_Her_Training import Gazebo_Env
from Train_Utils import soft_update, hard_update, OrnsteinUhlenbeckProcess

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--test', type=bool, default=False)
args = parser.parse_args()

if(torch.cuda.is_available()):
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

pc_name = "kla-100"

##############################################################################
#Hyperparameters
if (args.model_path == None):
    config = EasyDict({
        "start_epi"             : 0,
        "n_agents"              : 2,
        "n_targets"             : 1,
        "dim_obs"               : 12,
        "dim_act"               : 3,
        "dim_goal"              : 2,
        "batch_size"            : 3,
        "capacity"              : 400000,
        "lr_ac"                 : 0.00001,
        "lr_cr"                 : 0.0005,
        "gamma"                 : 0.95,
        "first_hidden_layer"    : 128,
        "second_hidden_layer"   : 64,
        "tau"                   : 0.01,  # for target network soft update
        "delay_step"            : 6,
        "n_episode"             : 500000,
        "episodes_before_train" : 5,
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
        "dim_goal"              : json_data["dim_goal"],
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
        "dim_laser"             : json_data["dim_laser"],
        "dim_laser_output"      : json_data["dim_laser_output"]
    })


##############################################################################################################
# MADDPG 
class MADDPG:
    def __init__(self, 
                n_agents, 
                dim_obs, dim_act, dim_laser, dim_goal, 
                batch_size, las_output, 
                first_hidden_layer, second_hidden_layer,
                capacity, episodes_before_train, 
                gamma, tau, lr_cr, lr_ac):

        self.actors = [Actor(dim_obs + dim_goal, dim_act, dim_laser, las_output, first_hidden_layer, second_hidden_layer) for i in range(n_agents)]
        self.critics = [Critic(n_agents, dim_obs + dim_goal, dim_act, dim_laser, las_output, first_hidden_layer, second_hidden_layer) for i in range(n_agents)]

        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = dim_act
        self.n_laser = dim_laser
        self.n_goal = dim_goal
        self.memory = ReplayMemory(capacity)
        self.batch_size = batch_size
        self.episodes_before_train = episodes_before_train

        self.GAMMA = gamma
        self.tau = tau

        if(args.test == False):
            self.var = [1.0 for i in range(n_agents)]
        else:
            self.var = [0.0 for i in range(n_agents)]

        self.critic_optimizer = [optim.Adam(x.parameters(), lr=lr_cr) for x in self.critics]
        self.actor_optimizer = [optim.Adam(x.parameters(), lr=lr_ac) for x in self.actors]

        for a, c, at, ct in zip(self.actors, self.critics, self.actors_target, self.critics_target):
            a.to(device)
            c.to(device)
            at.to(device)
            ct.to(device)

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

            # state_batch: batch_size x n_agents x dim_obs
            state_batch = torch.stack(batch.states).type(FloatTensor)
            laser_batch = torch.stack(batch.laser).type(FloatTensor)
            action_batch = torch.stack(batch.actions).type(FloatTensor)
            next_state_batch = torch.stack(batch.next_states).type(FloatTensor)
            next_laser_batch = torch.stack(batch.next_laser).type(FloatTensor)
            reward_batch = torch.stack(batch.rewards).type(FloatTensor)
            non_final_mask = ~torch.tensor(batch.done).type(BoolTensor)

            print(non_final_mask)

            # for current agent
            whole_state = state_batch.view(self.batch_size, -1)
            whole_laser = laser_batch.view(self.batch_size, -1)
            whole_action = action_batch.view(self.batch_size, -1)
            current_Q = self.critics[agent](whole_state, whole_action, whole_laser)

            # n_agents x batch x next_action
            next_action_batch = [self.actors_target[i](next_state_batch[:, i, :], next_laser_batch[:, i, :]) for i in range(self.n_agents)]
            next_action_batch = torch.stack(next_action_batch)

            # batch x n_agents x next_action
            next_action_batch = (next_action_batch.transpose(0, 1).contiguous())

            target_Q = torch.zeros(self.batch_size).type(FloatTensor)
            target_Q = self.critics_target[agent](
                next_state_batch.view(-1, self.n_agents * (self.n_states + self.n_goal)),
                next_action_batch.view(-1, self.n_agents * self.n_actions),
                next_laser_batch.view(-1, self.n_agents * self.n_laser)
            ).squeeze()

            # TD Target = r + gamma * target_Q
            # TD Target shape : batch_size x 1 (agent)
            # non_final_mask shape : batch_size x 1
            target_Q = (non_final_mask.unsqueeze(1) * target_Q.unsqueeze(1) * self.GAMMA) + (reward_batch[:, agent].unsqueeze(1))

            # Update Critic Network
            loss_Q = nn.MSELoss()(current_Q, target_Q.detach())

            self.critic_optimizer[agent].zero_grad()
            loss_Q.backward()
            self.critic_optimizer[agent].step()

            # Update Actor Network
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
    def select_action(self, state_batch, laser_batch):
        # state_batch: n_agents x state_dim
        actions = torch.zeros(
            self.n_agents,
            self.n_actions)
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        
        for i in range(self.n_agents):

            #  state_batch : nagents x state count
            sb = state_batch[i, :].detach()
            lb = laser_batch[i, :].detach()

            act = self.actors[i](sb.unsqueeze(0), lb.unsqueeze(0))
            act = act.squeeze(0)

            act += torch.from_numpy(
                np.random.randn(self.n_actions) * self.var[i]).type(FloatTensor)

            if self.episode_done > self.episodes_before_train and self.var[i] > 0.05:
                self.var[i] *= 0.999993
            act = torch.clamp(act, -1.0, 1.0)

            actions[i, :] = act

        self.steps_done += 1
        return actions
##############################################################################################################
# MADDPG Replay Buffer
Experience = namedtuple('Experience',
                        ('states', 'laser', 'actions', 'next_states', 'next_laser', 'rewards', 'goal', 'done'))

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

        self.PreFC = nn.Linear(las_dim, las_out_dim)
        self.FC1 = nn.Linear(obs_dim + las_out_dim + act_dim, hidden_dim)
        self.FC2 = nn.Linear(hidden_dim, second_hidden_dim)
        self.FC3 = nn.Linear(second_hidden_dim, 1)

    # obs: batch_size * obs_dim
    def forward(self, obs, acts, laser):
        result = F.relu(self.PreFC(laser))
        combined = torch.cat([obs, result, acts], 1)
        result = F.relu(self.FC1(combined))
        result = F.relu(self.FC2(result))
        return self.FC3(result)


class Actor(nn.Module):
    def __init__(self, dim_observation, dim_action, dim_laser, las_output = 10, hidden_layer=128, second_hidden_layer=64):
        super(Actor, self).__init__()
        self.FC1 = nn.Linear(dim_observation + las_output, hidden_layer)
        self.FC2 = nn.Linear(hidden_layer, second_hidden_layer)
        self.FC3 = nn.Linear(second_hidden_layer, dim_action)

        self.PreFC = nn.Linear(dim_laser, las_output)

    def forward(self, obs, laser):
        result = F.relu(self.PreFC(laser))
        combined = torch.cat([obs, result], 1)
        result = F.relu(self.FC1(combined))
        result = F.relu(self.FC2(result))
        result = torch.tanh(self.FC3(result)) * 0.5
        return result

##################################################################################
def main():
    env = Gazebo_Env(config.n_agents, config.n_targets, config.dim_laser)

    print_interval = 100
    future_k = 4
    score = 0

    maddpg = MADDPG(config.n_agents, 
                    config.dim_obs, 
                    config.dim_act, 
                    config.dim_laser, 
                    config.dim_goal,
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
        start_idx = 5
        end_idx = 10
        for idx in range(len(file_list)):
            if(Utils.isint(file_list[idx][start_idx:end_idx])):
                if(max_value < int(file_list[idx][start_idx:end_idx])):
                    max_index = idx
                    max_value = int(file_list[idx][start_idx:end_idx])
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
        # obs(numpy) : n_agents x n_state (2 x 12)
        obs, laser = env.reset()
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        if isinstance(laser, np.ndarray):
            laser = torch.from_numpy(laser).float()

        success = 0
        total_reward = 0.0
        n_step = 0

        past_obs_list, past_laser_list, past_action_list = [], [], []
        episode_trajectory = []
        done = False

        while not done:
            obs = obs.type(FloatTensor)
            laser = laser.type(FloatTensor)
            goal = torch.from_numpy(env.goal_state).type(FloatTensor)

            obs_goal = torch.cat((obs, goal), 1)

            # h_out_list : n_agents x num_output(tuple) x (num_layer * num_direction) x batch x hidden
            torch_action = maddpg.select_action(obs_goal, laser)
            action = torch_action.data.cpu()
            obs_, laser_, done, reward, _ = env.step(action.numpy().tolist())

            reward = torch.FloatTensor(reward).type(FloatTensor)
            obs_ = torch.from_numpy(obs_).float()
            laser_ = torch.from_numpy(laser_).float()

            total_reward += reward.sum()

            if (n_step >= config.delay_step):
                episode_trajectory.append(Experience(past_obs_list.pop(0), past_laser_list.pop(0), past_action_list.pop(0), obs_, laser_, reward, goal, done))

            past_obs_list.append(obs.data.cpu())
            past_laser_list.append(laser.data.cpu())
            past_action_list.append(action)

            obs = obs_
            laser = laser_
            rate.sleep()

            n_step += 1

        #########################################################
        # check static goal (absoulte value NOT relative value) #
        #########################################################
        episode_step = len(episode_trajectory)
        for t in range(episode_step):
            state, laser, action, next_state, next_laser, reward, goal, done = episode_trajectory[t]
            state_, next_state_ = torch.cat((state, goal.cpu()), 1), torch.cat((next_state, goal.cpu()), 1)
            maddpg.memory.push(state_, laser, action, next_state_, next_laser, reward, None, done)

            for _ in range(future_k):
                future = random.randint(t, episode_step - 1)
                new_goal = episode_trajectory[future].next_states[0, :2]

                new_reward = env.compute_reward(next_state, new_goal)
                new_reward = torch.FloatTensor(new_reward).type(FloatTensor)

                state_, next_state_ = torch.cat((state, new_goal), 1), torch.cat((next_state, new_goal), 1)
                maddpg.memory.push(state_, laser, action, next_state_, next_laser, new_reward, None, done)

        if(args.test == False):
            for i in range(5):
                c_loss, a_loss = maddpg.update_policy()
            
        score += total_reward
        maddpg.episode_done += 1

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
                ckpt_path = os.path.join(model_path, 'ckpt_'+ str(i_episode) + '_' + avg_score_str + '.pth')

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