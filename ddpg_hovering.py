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
from torch.distributions import Categorical
import collections

# Tensorboard
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--best_model', type=bool, default=False)
parser.add_argument('--load_model', type=bool, default=False)
args = parser.parse_args()

##################################################################################
# RL Part
##################################################################################
#Hyperparameters
lr_mu        = 0.0005
lr_q         = 0.001
gamma        = 0.99
batch_size   = 32
buffer_limit = 50000
tau          = 0.005 # for target network soft update

input_state = 13
output_state = 4

start_hidden_mu = 128
second_hidden_mu = 64

start_hidden_q = 64
second_hidden_q = 32

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0 
            done_mask_lst.append([done_mask])
        
        return torch.tensor(s_lst, dtype=torch.float).to('cuda'), torch.tensor(a_lst, dtype=torch.float).to('cuda'), \
                torch.tensor(r_lst, dtype=torch.float).to('cuda'), torch.tensor(s_prime_lst, dtype=torch.float).to('cuda'), \
                torch.tensor(done_mask_lst, dtype=torch.float).to('cuda')
    
    def size(self):
        return len(self.buffer)

class MuNet(nn.Module):
    def __init__(self):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(input_state, start_hidden_mu)
        self.fc2 = nn.Linear(start_hidden_mu, second_hidden_mu)
        self.fc_mu = nn.Linear(second_hidden_mu, output_state)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x)) * 0.5
        return mu

class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(input_state, start_hidden_q)
        self.fc_a = nn.Linear(output_state,start_hidden_q)
        self.fc_q = nn.Linear(start_hidden, second_hidden_q)
        self.fc_out = nn.Linear(second_hidden_q, 1)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1,h2], dim=1)
        q = F.relu(self.fc_q(cat))
        q = self.fc_out(q)
        return q

class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x
      
def train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer):
    s,a,r,s_prime,done_mask  = memory.sample(batch_size)
    
    target = r + gamma * q_target(s_prime, mu_target(s_prime)) * done_mask
    q_loss = F.smooth_l1_loss(q(s,a), target.detach())
    q_optimizer.zero_grad()
    q_loss.backward()
    q_optimizer.step()
    
    mu_loss = -q(s,mu(s)).mean() # That's all for the policy loss.
    mu_optimizer.zero_grad()
    mu_loss.backward()
    mu_optimizer.step()
    
def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

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

        self.rate_cmd = action
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
    
    memory = ReplayBuffer()

    q, q_target = QNet().to('cuda'), QNet().to('cuda')
    q_target.load_state_dict(q.state_dict())

    mu, mu_target = MuNet().to('cuda'), MuNet().to('cuda')
    mu_target.load_state_dict(mu.state_dict())

    mu_optimizer = optim.Adam(mu.parameters(), lr=lr_mu)
    q_optimizer  = optim.Adam(q.parameters(), lr=lr_q)
    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(output_state))

    score = 0.0
    epi_reward = 0.0
    print_interval = 10
    n_epi = 0

    if (args.load_model != None):
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

    Save_path = "model/" + str(year) + "_" + str(month) + "_" + str(day) + "_" + str(hour) + "_" + str(minute) + "_save_sca_model_" + \
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

            a = mu(torch.from_numpy(s).float())
            for i in range(output_state): 
                new_a.append(a.item()[i] + ou_noise()[i])
            s_prime, r, done, info = env.step(new_a)
            memory.put((s,new_a,r/100.0,s_prime,done))
            score += r
            epi_reward += r
            s = s_prime
            

            if done:
                break

        if memory.size() > 2000:
            for i in range(10):
                train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer)
                soft_update(mu, mu_target)
                soft_update(q,  q_target)

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