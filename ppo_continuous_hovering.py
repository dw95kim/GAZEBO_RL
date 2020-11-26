#!/usr/bin/env python
# ROS
import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Float32MultiArray, Int32
from nav_msgs.msg import Odometry
import tf

# Env
import os
import datetime
import numpy as np
import random
import math
import time
import argparse

# RL training
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.distributions import Categorical
import collections

# Tensorboard
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--load_model', type=bool, default=False)
args = parser.parse_args()

device = torch.device("cuda")

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

#Hyperparameters
learning_rate  = 0.0003
gamma           = 0.9
lmbda           = 0.9
eps_clip        = 0.2
K_epoch         = 10
rollout_len    = 3
buffer_size    = 30
minibatch_size = 32

scale_factor = 0.5

input_state = 13
output_state = 4
hidden_state = 128

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []
        self.delayed_data = []
        
        self.fc1   = nn.Linear(input_state, hidden_state)
        self.fc_mu = nn.Linear(hidden_state, output_state)
        self.fc_std  = nn.Linear(hidden_state, output_state)
        self.fc_v = nn.Linear(hidden_state, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.optimization_step = 0

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        mu = scale_factor * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
      
    def delay_data(self, delay_step):
        temp_s, temp_a, temp_r, temp_sprime, temp_prob_a, temp_done = [], [], [], [], [], []

        index = 0
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            if (index >= delay_step):
                temp_sprime.append(s_prime)
                temp_r.append([r])
                done_mask = 0 if done else 1
                temp_done.append([done_mask])

            if (index < len(self.data) - delay_step):
                temp_s.append(s)
                temp_a.append([a])
                temp_prob_a.append([prob_a])
            
            index += 1

        for i in range(len(temp_r)):
            self.delayed_data.append((temp_s[i], temp_a[i], temp_r[i], temp_sprime[i], temp_prob_a[i], temp_done[i]))

    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_batch, a_batch, r_batch, s_prime_batch, prob_a_batch, done_batch = [], [], [], [], [], []
        data = []

        for j in range(buffer_size):
            for i in range(minibatch_size):
                rollout = self.delayed_data.pop()
                s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []

                for transition in rollout:
                    s, a, r, s_prime, prob_a, done = transition
                    
                    s_lst.append(s)
                    a_lst.append(a)
                    r_lst.append([r])
                    s_prime_lst.append(s_prime)
                    prob_a_lst.append([prob_a])
                    done_mask = 0 if done else 1
                    done_lst.append([done_mask])

                s_batch.append(s_lst)
                a_batch.append(a_lst)
                r_batch.append(r_lst)
                s_prime_batch.append(s_prime_lst)
                prob_a_batch.append(prob_a_lst)
                done_batch.append(done_lst)
                    
            mini_batch = torch.tensor(s_batch, dtype=torch.float).to('cuda'), torch.tensor(a_batch, dtype=torch.float).to('cuda'), \
                          torch.tensor(r_batch, dtype=torch.float).to('cuda'), torch.tensor(s_prime_batch, dtype=torch.float).to('cuda'), \
                          torch.tensor(done_batch, dtype=torch.float).to('cuda'), torch.tensor(prob_a_batch, dtype=torch.float).to('cuda')
            data.append(mini_batch)

        return data

    def calc_advantage(self, data):
        data_with_adv = []
        for mini_batch in data:
            s, a, r, s_prime, done_mask, old_log_prob = mini_batch
            with torch.no_grad():
                td_target = r + gamma * self.v(s_prime) * done_mask
                delta = td_target - self.v(s)
            delta = delta.numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)
            data_with_adv.append((s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage))

        return data_with_adv

        
    def train_net(self):
        if len(self.delayed_data) == minibatch_size * buffer_size:
            data = self.make_batch()
            data = self.calc_advantage(data)

            for i in range(K_epoch):
                for mini_batch in data:
                    s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage = mini_batch

                    mu, std = self.pi(s, softmax_dim=1)
                    dist = Normal(mu, std)
                    log_prob = dist.log_prob(a)
                    ratio = torch.exp(log_prob - old_log_prob)  # a/b == exp(log(a)-log(b))

                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
                    loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target)

                    self.optimizer.zero_grad()
                    loss.mean().backward()
                    nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimization_step += 1


##################################################################################
# Env Part
##################################################################################
d2r = math.pi/180
r2d = 180/math.pi


# Helper Function
def distance_3d_vec(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)

def mag_3d_vec(a):
    return math.sqrt(a[0]**2 + a[1]**2 + a[2]**2)

def cal_angle_3d_vec(a, b):
    mag_a = mag_3d_vec(a)
    mag_b = mag_3d_vec(b)
    inner_prod = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
    if (mag_a * mag_b == 0):
        cos_value = 0
    else:
        cos_value = inner_prod / (mag_a * mag_b)
    return math.acos(cos_value)

# line_a : direction vector
# line_b : static point
def cal_dist_line_and_vec(line_a, line_b, vec):
    ref_line = np.array(np.array(vec) - np.array(line_b))
    mag_ref_line = mag_3d_vec(ref_line)
    norm_line_a = line_a / np.linalg.norm(line_a)
    cos_value = (norm_line_a[0]*ref_line[0] + norm_line_a[1]*ref_line[1] + norm_line_a[2]*ref_line[2])/mag_ref_line
    return mag_ref_line * math.sqrt(1 - cos_value**2)

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


class Gazebo_Env():
    def __init__(self):
        rospy.init_node('rl_node')
    
        self.Drone_pos = []
        self.pos_err = [0, 0, 0]
        self.pos_err_from_bf = [0, 0, 0]
        self.Drone_vel = [0, 0, 0]
        self.Drone_vel_from_bf = [0, 0, 0]
        self.Drone_angle = [] # Euler Angle

        # [r_rate, p_rate, y_rate, throttle]
        self.rate_cmd = []

        # Subscribe Node
        rospy.Subscriber('/racing/mavros/local_position/pose', PoseStamped, self.callback_pos)
        rospy.Subscriber('/racing/mavros/local_position/velocity_local', TwistStamped, self.callback_vel)

        # Publish Node : mission / goal x, y, z / goal yaw / max vel_x_y, vel_z
        self.goal_pub = rospy.Publisher("/racing/GoalAction", Float32MultiArray, queue_size=1)
        self.target_pub = rospy.Publisher("/racing/targret_pos", Float32MultiArray, queue_size=1)

        #############################
        # For Env and Training
        #############################
        self.goal = Float32MultiArray()
        self.goal.data = []

        self.target_pos_pub = Float32MultiArray()
        self.target_pos_pub.data = [0, 0, 0]

        self.state = []
        self.start_time = time.time()
        self.prev_callback_time = 0
        self.prev_step_time = 0

        # Debug Part
        self.inital_position = [0, 0, 40]
        self.inital_target_pos = [0, 0, 45] # +/- 5
        self.target_pos = []

        self.throttle_min_max = [0.4, 0.8]

        self.time_constrain = 100
        self.distance_over = 15
        self.alt_done = 20
        self.check_callback_Hz = False # Over 125Hz
        self.check_step_Hz = False # 156Hz (20.11.10)
        self.check_print_state = False

    def callback_vel(self, data):
        self.Drone_vel = np.array([data.twist.linear.x, data.twist.linear.y, data.twist.linear.z])

    def callback_pos(self, data):
        if (self.check_callback_Hz == True):
            now = time.time()
            print("Callback Hz : ", 1/(now-self.prev_callback_time))
            self.prev_callback_time = now 

        q = [
            data.pose.orientation.x,
            data.pose.orientation.y,
            data.pose.orientation.z,
            data.pose.orientation.w
        ]

        rot_matrix = QuaternionToDCM(q)

        self.Drone_pos = [data.pose.position.x, data.pose.position.y, data.pose.position.z]
        if (len(self.target_pos) != 0):
            self.pos_err = np.array(self.target_pos) - np.array(self.Drone_pos)
            self.target_pos_pub.data = [self.pos_err[0], self.pos_err[1], self.pos_err[2]]
            self.pos_err_from_bf = np.matmul(rot_matrix, self.pos_err)

        self.Drone_vel_from_bf = np.matmul(rot_matrix, self.Drone_vel)

        self.Drone_angle = np.array(list(tf.transformations.euler_from_quaternion(q)))

        if (self.check_print_state == True):
            print("Pose     : ", self.Drone_pos)
            print("Vel      : ", self.Drone_vel)
            print("Angle    : ", self.Drone_angle)
            print(" ")

    def make_state(self):
        self.state = np.concatenate((self.pos_err_from_bf, self.Drone_vel_from_bf, self.Drone_angle, self.rate_cmd), axis=0)

    def reset(self):
        x = (random.random()-0.5) * 10
        y = (random.random()-0.5) * 10
        z = (random.random()-0.5) * 10

        self.target_pos = [self.inital_target_pos[0] + x, self.inital_target_pos[1] + y, self.inital_target_pos[2] + z]
        self.rate_cmd = [0.0, 0.0, 0.0, 0.6]
        self.goal.data = []
        self.target_pos_pub.data = []

        self.state = []
        self.start_time = time.time()

        while (self.Drone_pos[2] < self.inital_position[2] - 0.1):
            self.goal.data = [1.0] + self.inital_position + [90.0, 0.0, 3.0]
            self.goal_pub.publish(self.goal)

        while (distance_3d_vec(self.Drone_pos, self.inital_position) > 1.5):
            self.goal.data = [3.0] + self.inital_position + [90.0, 2.0, 3.0]
            self.goal_pub.publish(self.goal)

        self.make_state()
        return self.state

    def step(self, action):
        now = time.time()
        if (self.check_step_Hz == True):
            print("Step Hz : ", 1/(now-self.prev_step_time))

        time_duration = now - self.prev_step_time        
        self.prev_step_time = now 

        if (action == 0):
            self.rate_cmd[0] += 0.005
        elif (action == 1):
            self.rate_cmd[0] -= 0.005
        elif (action == 2):
            self.rate_cmd[1] += 0.005
        elif (action == 3):
            self.rate_cmd[1] -= 0.005
        elif (action == 4):
            self.rate_cmd[2] += 0.005
        elif (action == 5):
            self.rate_cmd[2] -= 0.005
        elif (action == 6):
            self.rate_cmd[3] += 0.005
        else:
            self.rate_cmd[3] -= 0.005

        limit_cmd = 0.5

        self.rate_cmd[0] = clip(self.rate_cmd[0], -limit_cmd, limit_cmd)
        self.rate_cmd[1] = clip(self.rate_cmd[1], -limit_cmd, limit_cmd)
        self.rate_cmd[2] = clip(self.rate_cmd[2], -limit_cmd, limit_cmd)
        self.rate_cmd[3] = clip(self.rate_cmd[3], self.throttle_min_max[0], self.throttle_min_max[1])
        self.goal.data = [7] + self.rate_cmd + [1.0, 1.0]

        self.goal_pub.publish(self.goal)
        self.target_pub.publish(self.target_pos_pub)

        time.sleep(0.005)
        self.make_state()

        return self.state, self.get_reward(), self.is_gameover(), None
        
    def is_gameover(self):
        if (time.time() - self.start_time > self.time_constrain):
            print("Time Constrain")
            return True
        if (distance_3d_vec(self.target_pos, self.Drone_pos) > self.distance_over):
            print("distance Over")
            return True
        if (self.Drone_pos[2] < self.alt_done):
            print("Alt Done")
            return True

    def get_reward(self):
        reward = 0
        reward += self.distance_over - distance_3d_vec(self.target_pos, self.Drone_pos)
        return reward


def main():
    env = Gazebo_Env()
    
    model = PPO()
    model = model.to('cuda')

    rollout = []

    score = 0.0
    epi_reward = 0.0
    print_interval = 10
    n_epi = 0

    if (args.load_model == True):
        path = args.load_model
        print(path)

        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

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

    Save_path = "model/" + str(year) + "_" + str(month) + "_" + str(day) + "_" + str(hour) + "_" + str(minute) + "_save_model_" + \
                "first_hidden_" + str(start_hidden) + "_second_hidden_" + str(second_hidden) + "_input_" + str(input_state) + "_output_" + str(output_state)

    # tensorboard --logdir=runs
    writer = SummaryWriter()
    
    while(True):
        n_epi += 1
        s = env.reset()
        done = False

        while not done:
            if(done):
                s = env.reset()

            mu, std = model.pi(torch.from_numpy(s).float())
            dist = Normal(mu, std)
            a = dist.sample()
            log_prob = dist.log_prob(a)
            s_prime, r, done, info = env.step(a.item())

            rollout.append((s, a, r/10.0, s_prime, log_prob.item(), done))
            if len(rollout) == rollout_len:
                model.put_data(rollout)
                rollout = []

            s = s_prime
            score += r
            epi_reward += r
            if done:
                break

        model.train_net()

        writer.add_scalar("Reward function", epi_reward, n_epi)
        writer.add_scalar("Loss", model.epi_loss, n_epi)
        print("Reward : " + str(epi_reward))
        print(" ")
        epi_reward = 0

        if n_epi%print_interval==0 and n_epi!=0:
            model_path = os.path.join("/home/dwkim/RL_ws/src/3_rl/src", Save_path)
            if not os.path.isdir(model_path):
                os.makedirs(model_path)

            avg_score_str = str(int(score/print_interval))
            ckpt_path = os.path.join(model_path, 'ckpt_'+ avg_score_str + '.pth')

            torch.save(
                {
                'episode' : n_epi,
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict' : model.optimizer.state_dict(),
                }, ckpt_path)

            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0

if __name__ == '__main__':
    main()