
#!/usr/bin/env python
# ROS
import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import LaserScan, Imu
import tf

# Env
import numpy as np
import math
import time
import random

# Utils
from Utils import *

class Gazebo_Env():
    def __init__(self, n_agents, n_targets, n_laser):
        rospy.init_node('rl_node')
    
        self.n_agents = n_agents
        self.start_target_index = n_agents
        self.total_drone = n_agents + n_targets
        self.n_laser = n_laser

        # i --> Agent i th pose / velocity / angle
        self.pos_list = np.zeros((self.total_drone, 3))
        self.vel_list = np.zeros((self.total_drone, 3))
        self.angle_list = np.zeros((self.total_drone, 3))
        self.prev_angle_list = np.zeros((self.total_drone, 3))
        self.angular_vel_list = np.zeros(self.total_drone)

        self.laser_list = np.zeros((n_agents, n_laser))

        # i,j --> Agent i pos/vel from Agent j frame
        self.pos_err_jtoi_from_j_frame = np.zeros((self.total_drone, self.total_drone, 3))
        self.vel_err_jtoi_from_j_frame = np.zeros((self.total_drone, self.total_drone, 3))
        self.prev_target_pos_err = np.zeros((self.n_agents, 3))

        # [x_vel, y_vel, z_vel, throttle]
        self.vel_cmd_list = np.zeros((n_agents, 4))

        # Subscribe Node
        for i in range(self.total_drone):
            rospy.Subscriber('/UAV_' + str(i) + '/mavros/local_position/pose', PoseStamped, self.callback_pos, (i))
            rospy.Subscriber('/UAV_' + str(i) + '/mavros/local_position/velocity_local', TwistStamped, self.callback_vel, (i))
            rospy.Subscriber('/UAV_' + str(i) + '/scan', LaserScan, self.callback_laser, (i))

        # Publish Node : mission / goal x, y, z / goal yaw / max vel_x_y, vel_z
        self.pub_goal_list = []
        self.pub_input_list = []
        for i in range(self.total_drone):
            self.pub_goal_list.append(rospy.Publisher('/UAV_' + str(i) + '/GoalAction', Float32MultiArray, queue_size=1))
            
        for i in range(self.n_agents):
            self.pub_input_list.append(rospy.Publisher('/UAV_' + str(i) + '/check/input', Float32MultiArray, queue_size=1))

        #############################
        # For Env and Training
        #############################
        self.goal_list = []
        self.input_list = []
        for i in range(self.total_drone):
            self.goal_list.append(Float32MultiArray())
            
        for i in range(self.n_agents):
            self.input_list.append(Float32MultiArray())

        self.set_first_start_time = True

        # prev part
        self.state = []
        self.start_time = 0 # pose callback count # 30hz
        self.prev_target_pos = np.zeros(3)
        self.prev_target_distance = np.zeros(self.n_agents)

        # check hz part
        self.cur_time = 0
        self.prev_time = 0

        # Done Part
        self.capture_done = [0 for i in range(self.n_agents)]
        self.collision_done = [0 for i in range(self.n_agents)]
        self.out_field_done = [0 for i in range(self.n_agents)]
        self.is_detect = 0
        self.last_target_cmd_time = 0

        # Debug Part
        self.map = 2 # 1 : Hard Map, 2 : City Map
        self.inital_position = [np.array([3, 3, 10]), 
                                np.array([17, 17, 10]), 
                                np.array([25, 25, 10]),
                                np.array([30, 30, 10])]

        self.offset_list = [np.array([3, 3, 0]), 
                            np.array([17, 17, 0]), 
                            np.array([25, 25, 0]),
                            np.array([30, 30, 0])]

        self.start_angle = [0.0, 0.0, 0.0, 0.0]
        self.target_response_time = 1
        self.time_constrain = 30000.0
        self.min_distance = 1.5
        self.min_laser = -0.95
        self.target_manuver = 2 # 1 : Avoid Prob, 2 : Potential

        self.agent_fov = 180 # drgree
        self.reward_type = 2 # 1 : Sparse, 2 : Dense
        
        if (self.map == 2):
            self.map_size = [40, 40]
        else:
            self.map_size = [20, 20]

    def callback_laser(self, data, agents):
        sample_index = int(1081 / (self.n_laser - 1))
        temp_laser = []
        for i in range(self.n_laser):
            if (i != self.n_laser - 1):
                if (data.ranges[i * sample_index] == np.inf):
                    value = 1
                else:
                    value = (data.ranges[i * sample_index] - 10) / 10.0
            else:
                if (data.ranges[-1] == np.inf):
                    value = 1
                else:
                    value = (data.ranges[-1] - 10) / 10.0
            temp_laser.append(value)
        self.laser_list[agents] = np.array(temp_laser)

    def callback_vel(self, data, agents):
        self.vel_list[agents] = np.array([data.twist.linear.x, data.twist.linear.y, data.twist.linear.z])

    def callback_pos(self, data, agents):
        self.start_time += 1

        q = [
            data.pose.orientation.x,
            data.pose.orientation.y,
            data.pose.orientation.z,
            data.pose.orientation.w
        ]

        rot_matrix = QuaternionToDCM(q)

        self.pos_list[agents] = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z]) + np.array([self.inital_position[agents][0], self.inital_position[agents][1], 0])

        for i in range(self.total_drone):
            self.pos_err_jtoi_from_j_frame[i][agents] = np.matmul(rot_matrix, self.pos_list[i] - self.pos_list[agents])

        if (agents < self.start_target_index):
            self.prev_target_pos_err[agents] = np.matmul(rot_matrix, self.prev_target_pos - self.pos_list[agents])

        self.prev_angle_list[agents] = self.angle_list[agents]
        self.angle_list[agents] = np.array(list(tf.transformations.euler_from_quaternion(q))) / 2.0
        self.angular_vel_list[agents] = (self.angle_list[agents][2] - self.prev_angle_list[agents][2]) / 0.0333333 #30hz

        for i in range(self.total_drone):
            self.vel_err_jtoi_from_j_frame[i][agents] = np.matmul(rot_matrix, self.vel_list[i])
    
    def make_state(self):
        # target relative distance / yaw
        # own pos / x, y vel / y rate / x, y cmd / yaw cmd
        # friends relative distance / yaw
        state_list = [0 for i in range(self.n_agents)]
        for i in range(self.n_agents):
            distance_target = mag_2d_vec(self.pos_err_jtoi_from_j_frame[self.start_target_index][i][:2])/self.map_size[0]
            yaw_target = math.atan2(self.pos_err_jtoi_from_j_frame[self.start_target_index][i][1], self.pos_err_jtoi_from_j_frame[self.start_target_index][i][0])
            state_list[i] = [distance_target, yaw_target]
            state_list[i] = state_list[i] + ((self.pos_list[i] - self.map_size[0]/2)/self.map_size[0]/2).tolist()[:2] + [self.vel_err_jtoi_from_j_frame[i][i].tolist()[0], self.vel_err_jtoi_from_j_frame[i][i].tolist()[1]] + [self.angular_vel_list[i].tolist()] + \
                            [self.vel_cmd_list[i].tolist()[0], self.vel_cmd_list[i].tolist()[1], self.vel_cmd_list[i].tolist()[3]]
            
            for j in range(self.n_agents):
                if j != i:
                    distance_friends = mag_2d_vec(self.pos_err_jtoi_from_j_frame[j][i][:2])/self.map_size[0]
                    yaw_friends = math.atan2(self.pos_err_jtoi_from_j_frame[j][i][1], self.pos_err_jtoi_from_j_frame[j][i][0])
                    state_list[i] = state_list[i] + [distance_friends, yaw_friends]
            state_list[i] = np.array(state_list[i])

        # Mask Part --> Convert Zero Value
        # self.is_detect = [1, 1] # 0 : can not detect target drone / 1 : can detect target drone
        # for i in range(self.n_agents):
        #     if (self.angle_list[i][2] == 0):
        #         self.angle_list[i][2] += 1e-5

        #     heading_vector = [math.cos(self.angle_list[i][2]), math.sin(self.angle_list[i][2])]
        #     vector_itoj = self.pos_list[self.start_target_index][:2] - self.pos_list[i][:2]

        #     if (cal_angle_2d_vec(heading_vector, vector_itoj) > self.agent_fov / 2 * d2r):
        #         self.is_detect[i] = 0
        #     else:
        #         # Check Obstacle
        #         # Original Map
        #         if (self.map == 1):
        #             obs_pos_list = [[10, 10], [15, 10]]
        #             obs_rad_list = [2.5, 2.5]
        #             for j in range(len(obs_pos_list)):
        #                 obs_pos = obs_pos_list[j]
        #                 obs_rad = obs_rad_list[j]
        #                 if (ismasked(self.pos_list[i][:2], self.pos_list[self.start_target_index][:2], obs_pos, obs_rad)):
        #                     self.is_detect[i] = 0
        #                     break
        #         elif (self.map == 2):# Masking Map
        #             obs_pos_list = [[3.5, 8], [10, 10], [15, 3], [3, 17], [12, 18], [17, 9], [4, 3], [11, 6]]
        #             obs_rad_list = [1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5]
        #             for j in range(len(obs_pos_list)):
        #                 obs_pos = obs_pos_list[j]
        #                 obs_rad = obs_rad_list[j]
        #                 if (ismasked(self.pos_list[i][:2], self.pos_list[self.start_target_index][:2], obs_pos, obs_rad)):
        #                     self.is_detect[i] = 0
        #                     break
        #         elif (self.map == 3):
        #             obs_pos_list = [[16, 32.0], [10.0, 18.0], [20.0, 6.0], [36, 30.0], [36, 8.0]]
        #             obs_wh_list = [[20, 8], [12, 6], [16, 6], [4, 12], [4, 8]]

        #             for j in range(len(obs_pos_list)):
        #                 obs_pos = obs_pos_list[j]
        #                 obs_wh = obs_wh_list[j]
        #                 if (isboxmasked(self.pos_list[i][:2], self.pos_list[self.start_target_index][:2], obs_pos, obs_wh)):
        #                     self.is_detect[i] = 0
        #                     break

        # if (sum(self.is_detect) == 0):
        #     for i in range(self.n_agents):
        #         distance_target = mag_2d_vec(self.prev_target_pos_err[i][:2])/self.map_size[0]
        #         yaw_target = math.atan2(self.prev_target_pos_err[i][1], self.prev_target_pos_err[i][0])
        #         state_list[i][:2] = [distance_target, yaw_target]
        #         state_list[i][:2] = [-5, -5]
        # else:
        #     self.prev_target_pos = self.pos_list[self.start_target_index]

        self.state = np.row_stack(tuple([i for i in state_list]))

    def reset(self):
        self.set_first_start_time = True

        self.capture_done = [0 for i in range(self.n_agents)]
        self.collision_done = [0 for i in range(self.n_agents)]
        self.out_field_done = [0 for i in range(self.n_agents)]
        self.prev_target_distance = np.zeros(self.n_agents)

        zero_initial_position_list = []
        for i in range(self.total_drone):
            temp_position = self.offset_list[i] - self.inital_position[i]
            zero_initial_position_list.append([temp_position[0], temp_position[1], 10])

        new_initial_position_list = []
        for i in range(self.total_drone):
            new_initial_position_list.append([self.offset_list[i][0], self.offset_list[i][1], 10])

        # Moving Specific position
        while (True):
            for i in range(self.total_drone):
                self.goal_list[i].data = [3.0] + zero_initial_position_list[i] + [self.start_angle[i]] + [1.0, 1.0]
                self.pub_goal_list[i].publish(self.goal_list[i])
        
            distance_list = []
            for i in range(self.total_drone):
                distance_list.append(distance_3d_vec(self.pos_list[i], new_initial_position_list[i]))
            if (max(distance_list) < 2):
                rospy.sleep(3)
                break
        print("PASS Reset Position")

        self.make_state()
        return self.state, self.laser_list

    # action torch type
    # [[agent1_x_vel, agent1_y_vel, agent1_r_vel],
    # [agent2_x_vel, agent2_y_vel, agent2_r_vel]]
    def step(self, action):
        if (self.set_first_start_time == True):
            self.start_time = 0
            self.set_first_start_time = False

        # check hz
        if (self.set_first_start_time == False):
            self.cur_time = time.time()
            print( 1 / (self.cur_time - self.prev_time))
            self.prev_time = self.cur_time

        for i in range(self.n_agents):
            self.vel_cmd_list[i] = [0.5 + action[i][0], action[i][1], satmax(1.5*(self.inital_position[i][2] - self.pos_list[i][2]), 1) + 0.3*(0.0 - self.vel_list[i][2]), action[i][2]]
            self.goal_list[i].data = [7.0] + self.vel_cmd_list[i].tolist() + [0.0, 0.0]

        if (time.time() - self.last_target_cmd_time > self.target_response_time):
            if (self.map == 1):
                occupancy_grid = np.zeros((self.map_size[0] + 2, self.map_size[0] + 2))
                obstacle_pos_big = [[3.5, 8], [10, 10], [15, 3]]
                obstacle_pos_small = [[3, 17], [12, 18], [17, 9], [4, 3], [11, 6]]

                tracking_drone_pos = [self.pos_list[0][:2], self.pos_list[1][:2]]
                eps = 0.000001

                masked = [0 for i in range(self.n_agents)]
                obs_pos_list = [[3.5, 8], [10, 10], [15, 3], [3, 17], [12, 18], [17, 9], [4, 3], [11, 6]]
                obs_rad_list = [1.5, 1.5, 1.5, 1, 1, 1, 1, 1]
                
                for i in range(len(obs_pos_list)):
                    obs_pos = obs_pos_list[i]
                    obs_rad = obs_rad_list[i]
                    
                    for j in range(self.n_agents):
                        if (ismasked(self.pos_list[i][:2]), self.pos_list[self.start_target_index][:2], obs_pos, obs_rad):
                            masked[j] = 1

                for i in range(self.map_size[0] + 2):
                    for j in range(self.map_size[0] + 2):
                        pix = [i, j]

                        drone_force = 0
                        for k in range(self.n_agents):
                            if (masked[k] == 0):
                                drone_force += 20 / (distance_2d_vec(pix, tracking_drone_pos[k]) + eps)

                        if (i < self.map_size[0]/2 and j < self.map_size[1]/2):
                            field_force = 10 / (distance_2d_vec(pix, [0, j]) + eps) + 10 / (distance_2d_vec(pix, [i, 0]) + eps)
                        elif (i < self.map_size[0]/2 and j >= self.map_size[1]/2):
                            field_force = 10 / (distance_2d_vec(pix, [0, j]) + eps) + 10 / (distance_2d_vec(pix, [i, self.map_size[1]]) + eps)
                        elif (i >= self.map_size[0]/2 and j < self.map_size[1]/2):
                            field_force = 10 / (distance_2d_vec(pix, [self.map_size[0], j]) + eps) + 10 / (distance_2d_vec(pix, [i, 0]) + eps)
                        else:
                            field_force = 10 / (distance_2d_vec(pix, [self.map_size[0], j]) + eps) + 10 / (distance_2d_vec(pix, [i, self.map_size[1]]) + eps)

                        obstacle_force = 0
                        for k in range(len(obstacle_pos_big)):
                            obstacle_force += 5 / (distance_2d_vec(pix, obstacle_pos_big[k]) + eps)
                        
                        for k in range(len(obstacle_pos_small)):
                            obstacle_force += 3 / (distance_2d_vec(pix, obstacle_pos_small[k]) + eps)

                        occupancy_grid[i, j] = drone_force + field_force + obstacle_force

                cur_pix = np.array([int(round(self.pos_list[self.start_target_index][0])), int(round(self.pos_list[self.start_target_index][1]))])
                if (cur_pix[0] == 0): cur_pix[0] += 1
                if (cur_pix[0] == self.map_size[0] + 1): cur_pix[0] -= 1
                if (cur_pix[1] == 0): cur_pix[1] += 1
                if (cur_pix[1] == self.map_size[0] + 1): cur_pix[1] -= 1

                local_occupancy = occupancy_grid[cur_pix[0] - 1 : cur_pix[0] + 2, cur_pix[1] - 1 : cur_pix[1] + 2]

                local_min_point = np.where(local_occupancy == local_occupancy.min())
                local_min_point_list = [int(local_min_point[0]) - 1, int(local_min_point[1]) - 1]
                min_point = local_min_point_list + cur_pix

                heading_vector = (np.array(min_point) - self.pos_list[self.start_target_index][:2])
                heading_angle = math.atan(heading_vector[1]/heading_vector[0])

                min_point = (np.array(min_point) - self.inital_position[self.start_target_index][:2]).tolist()
                self.goal_list[self.start_target_index].data = [8.0] + min_point + [10, heading_angle] + [1.0, 1.0]
            elif (self.map == 2):
                occupancy_grid = np.zeros((self.map_size[0] + 2, self.map_size[0] + 2))
                obs_pos_list = [[16, 32.0], [10.0, 18.0], [20.0, 6.0], [36, 30.0], [36, 8.0]]
                obs_wh_list = [[11, 5], [7, 4], [9, 4], [3, 7], [3, 6]]

                tracking_drone_pos = [self.pos_list[0][:2], self.pos_list[1][:2]]
                eps = 0.000001

                masked = [0, 0]
                for i in range(len(obs_pos_list)):
                    obs_pos = obs_pos_list[i]
                    obs_wh = obs_wh_list[i]
                    if (isboxmasked(self.pos_list[0][:2], self.pos_list[self.start_target_index][:2], obs_pos, obs_wh)):
                        masked[0] = 1
                    if (isboxmasked(self.pos_list[1][:2], self.pos_list[self.start_target_index][:2], obs_pos, obs_wh)):
                        masked[1] = 1

                for i in range(self.map_size[0] + 2):
                    for j in range(self.map_size[0] + 2):
                        pix = [i, j]

                        drone_force = 0
                        if (masked[0] == 0):
                            drone_force += 20 / (distance_2d_vec(pix, tracking_drone_pos[0]) + eps)
                        if (masked[1] == 0):
                            drone_force += 20 / (distance_2d_vec(pix, tracking_drone_pos[1]) + eps)

                        if (i < self.map_size[0]/2 and j < self.map_size[1]/2):
                            field_force = 10 / (distance_2d_vec(pix, [0, j]) + eps) + 10 / (distance_2d_vec(pix, [i, 0]) + eps)
                        elif (i < self.map_size[0]/2 and j >= self.map_size[1]/2):
                            field_force = 10 / (distance_2d_vec(pix, [0, j]) + eps) + 10 / (distance_2d_vec(pix, [i, self.map_size[1]]) + eps)
                        elif (i >= self.map_size[0]/2 and j < self.map_size[1]/2):
                            field_force = 10 / (distance_2d_vec(pix, [self.map_size[0], j]) + eps) + 10 / (distance_2d_vec(pix, [i, 0]) + eps)
                        else:
                            field_force = 10 / (distance_2d_vec(pix, [self.map_size[0], j]) + eps) + 10 / (distance_2d_vec(pix, [i, self.map_size[1]]) + eps)

                        obstacle_force = 0
                        for k in range(len(obs_pos_list)):
                            obs_pos = obs_pos_list[k]
                            obs_wh = obs_wh_list[k]
                            obstacle_force += generate_box_force(pix, obs_pos, obs_wh, 1)

                        occupancy_grid[i, j] = drone_force + field_force + obstacle_force

                cur_pix = np.array([int(round(self.pos_list[self.start_target_index][0])), int(round(self.pos_list[self.start_target_index][1]))])
                if (cur_pix[0] == 0): cur_pix[0] += 1
                if (cur_pix[0] == self.map_size[0] + 1): cur_pix[0] -= 1
                if (cur_pix[1] == 0): cur_pix[1] += 1
                if (cur_pix[1] == self.map_size[0] + 1): cur_pix[1] -= 1

                local_occupancy = occupancy_grid[cur_pix[0] - 1 : cur_pix[0] + 2, cur_pix[1] - 1 : cur_pix[1] + 2]

                local_min_point = np.where(local_occupancy == local_occupancy.min())
                if (len(local_min_point[0]) != 1):
                    x_ind = int(local_min_point[0][0])
                else:
                    x_ind = int(local_min_point[0])
                
                if (len(local_min_point[1]) != 1):
                    y_ind = int(local_min_point[1][0])
                else:
                    y_ind = int(local_min_point[1])

                local_min_point_list = [int(local_min_point[0]) - 1, int(local_min_point[1]) - 1]
                min_point = local_min_point_list + cur_pix

                heading_vector = (np.array(min_point) - self.pos_list[self.start_target_index][:2])
                heading_angle = math.atan(heading_vector[1]/heading_vector[0])

                min_point = (np.array(min_point) - self.inital_position[self.start_target_index][:2]).tolist()
                self.goal_list[self.start_target_index].data = [8.0] + min_point + [10, heading_angle] + [1.0, 1.0]
        self.last_target_cmd_time = time.time()

        # Publish part
        if(self.target_manuver > 0):
            for i in range(self.total_drone):
                self.pub_goal_list[i].publish(self.goal_list[i])
        else:
            for i in range(self.n_agents):
                self.pub_goal_list[i].publish(self.goal_list[i])

        for i in range(self.n_agents):
            self.input_list[i].data = self.state[i].tolist() + self.laser_list[i].tolist()
            self.pub_input_list[i].publish(self.input_list[i])

        # make state part
        self.make_state()

        return self.state, self.laser_list, self.is_gameover(), self.get_reward(), None

    def is_gameover(self):
        if (self.map == 1):
            poll_center_list = [[3.5, 8], [10, 10], [15, 3], [3, 17], [12, 18], [17, 9], [4, 3], [11, 6]]
            poll_radius_list = [1.5, 1.5, 1.5, 1, 1, 1, 1, 1]
            for i in range(self.total_drone):
                temp_position = [random.random() * self.map_size[0], random.random() * self.map_size[0], 0.0]
                if (i == 1):
                    while(inpollobstacle(temp_position[:2], poll_center_list, poll_radius_list) or distance_2d_vec(temp_position[:2], self.offset_list[0][:2]) < self.min_distance):
                        temp_position = [random.random() * self.map_size[0], random.random() * self.map_size[0], 0.0]
                elif (i == 2):
                    while(inpollobstacle(temp_position[:2], poll_center_list, poll_radius_list) or distance_2d_vec(temp_position[:2], self.offset_list[0][:2]) < self.min_distance or distance_2d_vec(temp_position[:2], self.offset_list[1][:2]) < self.min_distance):
                        temp_position = [random.random() * self.map_size[0], random.random() * self.map_size[0], 0.0]
                elif (i == 3):
                    while(inpollobstacle(temp_position[:2], poll_center_list, poll_radius_list) or distance_2d_vec(temp_position[:2], self.offset_list[0][:2]) < self.min_distance or distance_2d_vec(temp_position[:2], self.offset_list[1][:2]) < self.min_distance or distance_2d_vec(temp_position[:2], self.offset_list[2][:2]) < self.min_distance):
                        temp_position = [random.random() * self.map_size[0], random.random() * self.map_size[0], 0.0]
                else:
                    while(inpollobstacle(temp_position[:2], poll_center_list, poll_radius_list)):
                        temp_position = [random.random() * self.map_size[0], random.random() * self.map_size[0], 0.0]

                self.offset_list[i] = np.array(temp_position)
        elif (self.map == 2):
            obs_pos_list = [[16, 32.0], [10.0, 18.0], [20.0, 6.0], [36, 30.0], [36, 8.0]]
            obs_wh_list = [[12, 5], [8, 4], [10, 4], [4, 7], [4, 6]]            
            for i in range(self.total_drone):
                temp_position = [random.random() * self.map_size[0], random.random() * self.map_size[0], 0.0]
                if (i == 1):
                    while (inobstaclelist(temp_position[:2], obs_pos_list, obs_wh_list) or distance_2d_vec(temp_position[:2], self.offset_list[0][:2]) < self.min_distance):
                        temp_position = [random.random() * self.map_size[0], random.random() * self.map_size[0], 0.0]
                elif (i == 2):
                    while (inobstaclelist(temp_position[:2], obs_pos_list, obs_wh_list) or distance_2d_vec(temp_position[:2], self.offset_list[0][:2]) < self.min_distance or distance_2d_vec(temp_position[:2], self.offset_list[1][:2]) < self.min_distance):
                        temp_position = [random.random() * self.map_size[0], random.random() * self.map_size[0], 0.0]
                elif (i == 3):
                    while (inobstaclelist(temp_position[:2], obs_pos_list, obs_wh_list) or distance_2d_vec(temp_position[:2], self.offset_list[0][:2]) < self.min_distance or distance_2d_vec(temp_position[:2], self.offset_list[1][:2]) < self.min_distance or distance_2d_vec(temp_position[:2], self.offset_list[2][:2]) < self.min_distance):
                        temp_position = [random.random() * self.map_size[0], random.random() * self.map_size[0], 0.0]
                else:
                    while (inobstaclelist(temp_position[:2], obs_pos_list, obs_wh_list)):
                        temp_position = [random.random() * self.map_size[0], random.random() * self.map_size[0], 0.0]

                self.offset_list[i] = np.array(temp_position)

        zero_initial_position_list = []
        for i in range(self.total_drone):
            temp_position = self.offset_list[i] - self.inital_position[i]
            zero_initial_position_list.append([temp_position[0], temp_position[1], 10])

        if (self.start_time > self.time_constrain):
            print("Time Constrain")

            for i in range(self.total_drone):
                self.goal_list[i].data = [3.0] + zero_initial_position_list[i] + [self.start_angle[i]] + [1.0, 1.0]
                self.pub_goal_list[i].publish(self.goal_list[i])
            return True

        for i in range(self.start_target_index, self.total_drone):
            for j in range(self.n_agents):
                if(mag_3d_vec(self.pos_err_jtoi_from_j_frame[i][j]) < self.min_distance):
                    self.capture_done[j] = 1
                    print("Capture Success")

                    for t in range(self.total_drone):
                        self.goal_list[t].data = [3.0] + zero_initial_position_list[t] + [self.start_angle[t]] + [1.0, 1.0]
                        self.pub_goal_list[t].publish(self.goal_list[t])
                    return True

        for i in range(self.start_target_index):
            for j in range(self.n_laser):
                if (self.laser_list[i][j] < self.min_laser):
                    self.collision_done[i] = 1
                    print("Collision Warning with Obstacle by laser")

                    for t in range(self.total_drone):
                        self.goal_list[t].data = [3.0] + zero_initial_position_list[t] + [self.start_angle[t]] + [1.0, 1.0]
                        self.pub_goal_list[t].publish(self.goal_list[t])
                    return True

        distance_drone_list = []
        for i in range(self.start_target_index):
            for j in range(self.start_target_index):
                if (i != j):
                    distance_drone_list.append(mag_3d_vec(self.pos_err_jtoi_from_j_frame[i][j]))
        
        if (min(distance_drone_list) < self.min_distance):
            for i in range(self.n_agents):
                self.collision_done[i] = 1
            print("Collision Warning with Tracking Drone")
            
            for i in range(self.total_drone):
                self.goal_list[i].data = [3.0] + zero_initial_position_list[i] + [self.start_angle[i]] + [1.0, 1.0]
                self.pub_goal_list[i].publish(self.goal_list[i])
            return True

        for i in range(self.n_agents):
            if (self.pos_list[i][0] < -5 or self.pos_list[i][0] > self.map_size[0] + 5 or self.pos_list[i][1] < -5 or self.pos_list[i][1] > self.map_size[0] + 5):
                self.out_field_done[i] = 1
                print("Off field")

                for t in range(self.total_drone):
                    self.goal_list[t].data = [3.0] + zero_initial_position_list[t] + [self.start_angle[t]] + [1.0, 1.0]
                    self.pub_goal_list[t].publish(self.goal_list[t])
                return True
        return False