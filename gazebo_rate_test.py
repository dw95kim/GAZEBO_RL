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
learning_rate = 0.001
gamma         = 0.99
lmbda         = 0.95
eps_clip      = 0.2
K_epoch       = 20
T_horizon     = 2000000

#X Env
input_state = 13
output_state = 8
start_hidden = 128
second_hidden = 64

device = torch.device("cuda")

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []

        self.fc1        = nn.Linear(input_state, start_hidden)
        self.lstm       = nn.LSTM(start_hidden, second_hidden)
        self.fc_pi      = nn.Linear(second_hidden, output_state)
        self.fc_v       = nn.Linear(second_hidden, 1)
        self.optimizer  = optim.Adam(self.parameters(), lr=learning_rate)

        self.epi_loss   = 0

    def pi(self, x, hidden):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, start_hidden)
        x, lstm_hidden = self.lstm(x, hidden)
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=2)
        return prob, lstm_hidden
    
    def v(self, x, hidden):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, start_hidden)
        x, lstm_hidden = self.lstm(x, hidden)
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, h_in_lst, h_out_lst, done_lst = [], [], [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, h_in, h_out, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            h_in_lst.append(h_in)
            h_out_lst.append(h_out)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
        
        s,a,r,s_prime,done_mask,prob_a = torch.tensor(s_lst, dtype=torch.float).to('cuda'), torch.tensor(a_lst), \
                                         torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float).to('cuda'), \
                                         torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s,a,r,s_prime, done_mask, prob_a, h_in_lst[0], h_out_lst[0]
        
    def train_net(self):
        s,a,r,s_prime,done_mask, prob_a, (h1_in, h2_in), (h1_out, h2_out) = self.make_batch()
        first_hidden  = (h1_in.detach(), h2_in.detach())
        second_hidden = (h1_out.detach(), h2_out.detach())

        for i in range(K_epoch):
            v_prime = self.v(s_prime, second_hidden).squeeze(1)
            td_target = r.to('cuda') + gamma * v_prime * done_mask.to('cuda')
            v_s = self.v(s, first_hidden).squeeze(1)
            delta = td_target - v_s
            delta = delta.to('cpu')
            delta = delta.detach().numpy()
            
            advantage_lst = []
            advantage = 0.0
            for item in delta[::-1]:
                advantage = gamma * lmbda * advantage + item[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float).to('cuda')

            pi, _ = self.pi(s, first_hidden)
            pi_a = pi.squeeze(1).gather(1,a.to('cuda'))
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a.to('cuda')))  # a/b == log(exp(a)-exp(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            # td_target = td_target.to('cpu')
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(v_s, td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward(retain_graph=True)
            self.optimizer.step()

            self.epi_loss = loss.mean().data

##################################################################################
# Env Part
##################################################################################
d2r = math.pi/180
r2d = 180/math.pi

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

class Gazebo_Env():
    def __init__(self):
        rospy.init_node('rl_node')
    
        self.Drone_pos = []
        self.pos_err = []

        self.Drone_vel = []
        self.Drone_angle = [] # Euler Angle
        self.prev_Drone_angle = []
        self.Drone_angle_rate = []

        # [r_rate, p_rate, y_rate, throttle]
        self.rate_cmd = [0.0, 0.0, 0.0, 0.5]

        # Subscribe Node
        rospy.Subscriber('/racing/mavros/local_position/odom', Odometry, self.callback_pos)

        # Publish Node : mission / goal x, y, z / goal yaw / max vel_x_y, vel_z
        self.goal_pub = rospy.Publisher("/racing/GoalAction", Float32MultiArray, queue_size=1)
        self.target_pub = rospy.Publisher("/racing/targret_pos", Float32MultiArray, queue_size=1)

        #############################
        # For Env and Training
        #############################
        self.goal = Float32MultiArray()
        self.goal.data = []

        self.target_pos_pub = Float32MultiArray()
        self.target_pos_pub.data = []

        self.state = []
        self.start_time = time.time()
        self.prev_callback_time = 0
        self.prev_step_time = 0

        self.rl_start = False

        # log 
        self.file = None
        self.epi = 0

        # Debug Part
        self.inital_position = [0, 0, 30]
        self.target_pos = [0, 0, 35] # +/- 5

        self.throttle_min_max = [0.3, 0.7]

        self.time_constrain = 100
        self.distance_over = 15
        self.alt_done = 20
        self.check_callback_Hz = False # Over 20Hz ~ 30Hz
        self.check_step_Hz = True
        self.check_print_state = False

        self.delay_step = 5

        self.roll_rate_test = 1.0
        self.pitch_rate_test = 0.0
        self.yaw_rate_test = 0.0

    def callback_pos(self, data):
        now = time.time()
        if (self.check_callback_Hz == True):
            print("Callback Hz : ", 1/(now-self.prev_callback_time))

        time_duration = now - self.prev_callback_time
        self.prev_callback_time = now 

        self.Drone_pos = [data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z]
        self.pos_err = np.array(self.target_pos) - np.array(self.Drone_pos)

        self.Drone_vel = np.array([data.twist.twist.linear.x, data.twist.twist.linear.y, data.twist.twist.linear.z])

        q = (
                data.pose.pose.orientation.x,
                data.pose.pose.orientation.y,
                data.pose.pose.orientation.z,
                data.pose.pose.orientation.w
            )

        self.prev_Drone_angle = self.Drone_angle
        self.Drone_angle = np.array(list(tf.transformations.euler_from_quaternion(q)))

        if (len(self.prev_Drone_angle) != 0 and time_duration != 0):
            roll_rate = (self.Drone_angle[0] - self.prev_Drone_angle[0]) / time_duration
            pitch_rate = (self.Drone_angle[1] - self.prev_Drone_angle[1]) / time_duration
            yaw_rate = (self.Drone_angle[2] - self.prev_Drone_angle[2]) / time_duration
        else:
            roll_rate, pitch_rate, yaw_rate = 0, 0, 0

        self.Drone_angle_rate = [roll_rate, pitch_rate, yaw_rate]

        if (self.check_print_state == True):
            print("Pose     : ", self.Drone_pos)
            print("Vel      : ", self.Drone_vel)
            print("Angle    : ", self.Drone_angle)
            print("Rate     : ", self.Drone_angle_rate)

        if (self.file != None):
            line = [now] + self.Drone_angle_rate + [self.rl_start] 
            self.file.write(str(line) + "\n")

    def make_state(self):
        self.state = np.concatenate((self.pos_err, self.Drone_vel, self.Drone_angle, self.rate_cmd), axis=0)

    def reset(self):
        self.epi += 1
        file_path = "./rate_" + str(self.epi) + ".txt"

        self.file = open(file_path, "w")

        x = (random.random()-0.5) * 10
        y = (random.random()-0.5) * 10
        z = (random.random()-0.5) * 10

        self.target_pos = [self.target_pos[0] + x, self.target_pos[1] + y, self.target_pos[2] + z]
        self.rate_cmd = [0.0, 0.0, 0.0, 0.5]
        self.goal.data = []
        self.target_pos_pub.data = []

        self.state = []
        self.start_time = time.time()
        self.prev_callback_time = 0
        self.prev_step_time = 0

        self.rl_start = False

        while (self.Drone_pos[2] < self.inital_position[2] - 0.1):
            self.goal.data = [1.0] + self.inital_position + [90.0, 0.0, 2.0]
            self.goal_pub.publish(self.goal)

        while (distance_3d_vec(self.Drone_pos, self.inital_position) > 1.5):
            self.goal.data = [3.0] + self.inital_position + [90.0, 1.0, 2.0]
            self.goal_pub.publish(self.goal)

        self.make_state()
        return self.state

    def step(self, action):
    #######################################
    # make goal message
    #######################################
        now = time.time()
        if (self.check_step_Hz == True):
            print("Step Hz : ", 1/(now-self.prev_step_time))

        if (self.prev_step_time == 0):
            self.rl_start = True

        self.prev_step_time = now 

        if (action == 0):
            self.rate_cmd[0] += 0.01
        elif (action == 1):
            self.rate_cmd[0] -= 0.01
        elif (action == 2):
            self.rate_cmd[1] += 0.01
        elif (action == 3):
            self.rate_cmd[1] -= 0.01
        elif (action == 4):
            self.rate_cmd[2] += 0.01
        elif (action == 5):
            self.rate_cmd[2] -= 0.01
        elif (action == 6):
            self.rate_cmd[3] += 0.01
        else:
            self.rate_cmd[3] -= 0.01

        # self.rate_cmd[3] = clip(self.rate_cmd[3], self.throttle_min_max[0], self.throttle_min_max[1])
        # self.goal.data = [7] + self.rate_cmd + [1.0, 1.0]

        self.rate_cmd = [self.roll_rate_test, self.pitch_rate_test, self.yaw_rate_test, 0.5]
        self.goal.data = [7] + self.rate_cmd + [1.0, 1.0]

        self.goal_pub.publish(self.goal)
        self.target_pub.publish(self.target_pos_pub)

        time.sleep(0.0333)
        self.make_state()

        return self.state, self.get_reward(), self.is_gameover(), None
        
    def is_gameover(self):

        if (time.time() - self.start_time > self.time_constrain):
            print("Time Constrain")
            self.file.close()
            self.file = None
            return True
        if (distance_3d_vec(self.target_pos, self.Drone_pos) > self.distance_over):
            print("distance Over")
            self.file.close()
            self.file = None
            return True
        if (self.Drone_pos[2] < self.alt_done):
            print("Alt Done")
            self.file.close()
            self.file = None
            return True

    def get_reward(self):
        reward = 0
        reward += 20 - distance_3d_vec(self.target_pos, self.Drone_pos)
        return reward


def main():
    env = Gazebo_Env()
    
    model = PPO()
    model = model.to('cuda')

    score = 0.0
    epi_reward = 0.0
    print_interval = 10
    n_epi = 0

    if (args.best_model == True):
        path = ''
        print(path)

        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    elif (args.load_model == True):
        path = "./model/"
        file_list = os.listdir(path)
        last_folder = file_list[-1]
        
        path = "./model/" + last_folder
        file_list = os.listdir(path)
        max_index = 0
        max_value = int(file_list[0][5:-4])
        for idx in range(len(file_list)):
            if(max_value < int(file_list[idx][5:-4])):
                max_index = idx
                max_value = int(file_list[idx][5:-4])
        last_file = file_list[max_index]

        load_PATH = "model/" + last_folder + "/" + last_file
        print(load_PATH)

        checkpoint = torch.load(load_PATH)
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

    writer = SummaryWriter()
    
    while(True):
        n_epi += 1
        h_out = (torch.zeros([1, 1, second_hidden], dtype=torch.float, device = device), torch.zeros([1, 1, second_hidden], dtype=torch.float, device = device))
        s = env.reset()
        done = False

        s_history = [s]

        while not done:
            if(done):
                s = env.reset()

            h_in = h_out
            temp_s = torch.from_numpy(np.array(s)).float()
            prob, h_out = model.pi(temp_s.to('cuda'), h_in)
            prob = prob.view(-1)
            m = Categorical(prob)
            a = m.sample().item()

            s_prime, r, done, info = env.step(a)

            # delay step
            s_history.append(s_prime)

            model.put_data((s, a, r/100.0, s_prime, prob[a].item(), h_in, h_out, done))
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
            model_path = os.path.join("/home/usrg-asus/RL_ws/src/3_rl/src", Save_path)
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