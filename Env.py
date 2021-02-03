
#!/usr/bin/env python
# ROS
import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import LaserScan
import tf

# Env
import numpy as np
import math
import time

# Utils
from Utils import *

class Gazebo_Env():
    def __init__(self, n_agents, n_targets, n_laser):
        rospy.init_node('rl_node')
    
        # n_agents : 2, n_target : 1
        # [0, 1, 2] --> [agent, agent, target] --> start_target index = 2 = n_agents
        self.n_agents = n_agents
        self.start_target_index = n_agents
        self.total_drone = n_agents + n_targets
        self.n_laser = n_laser

        # i --> Agent i th pose / velocity / angle
        self.pos_list = np.zeros((self.total_drone, 3))
        self.vel_list = np.zeros((self.total_drone, 3))
        self.angle_list = np.zeros((self.total_drone, 3))
        
        # i,j --> Agent i pos/vel from Agent j frame
        self.pos_err_jtoi_from_j_frame = np.zeros((self.total_drone, self.total_drone, 3))
        self.vel_err_jtoi_from_j_frame = np.zeros((self.total_drone, self.total_drone, 3))

        # [x_vel, y_vel, z_vel, throttle]
        self.vel_cmd_list = np.zeros((n_agents, 4))

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
        for i in range(n_agents + n_targets):
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

        # Target Drone part
        self.detected_drone = [0, 0]
        self.last_detect_time = 0

        # Debug Part
        self.map_size = [20, 20]
        self.inital_position = [np.array([3, 3, 10]), np.array([17, 17, 10]), np.array([10, 2.5, 10])]
        self.offset_list = [np.array([3, 3, 0]), np.array([17, 17, 0]), np.array([10, 2.5, 0])]
        self.start_angle = [0.0, 3.141592, -1.57079]
        self.time_constrain = 100
        self.min_distance = 1.5
        self.Off_field_constrain = 100
        self.agent_fov = 120 # drgree
        self.target_manuver = 2 # 1 : Circle , 2 : Avoiding Alg with eyesight , 3 : Avoiding Alg without eyesight
        self.check_callback_Hz = False 
        self.check_step_Hz = False
        self.check_print_state = False
        self.write_command_file = False
        self.laser_list = np.zeros((n_agents, n_laser))
        self.reward_type = 2 # 1 : Sparse, 2 : Dense

    def callback_laser(self, data, agents):
        sample_index = int(1081 / self.n_laser)
        temp_laser = []
        for i in range(self.n_laser):
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
        ############################################################################
        # target pos, target vel, target yaw, own pos, own vel, own yaw, own vel cmd, friends pos, friends vel, friends yaw, 
        # 3 3 1 / 3 3 1 4 / 3 3 1 ==> 25
        # state_list = [0 for i in range(self.n_agents)]
        # for i in range(self.n_agents):
        #     state_list[i] = self.pos_list[self.start_target_index].tolist() + self.vel_list[self.start_target_index].tolist() + [self.angle_list[self.start_target_index][2]]
        #     state_list[i] = state_list[i] + self.pos_list[i].tolist() + self.vel_list[i].tolist() + [self.angle_list[i][2]] + self.vel_cmd_list[i].tolist()
        #     for j in range(self.n_agents):
        #         if j != i:
        #             state_list[i] = state_list[i] + self.pos_list[j].tolist() + self.vel_list[j].tolist() + [self.angle_list[j][2]]
        #     state_list[i] = np.array(state_list[i])

        # # Mask Part --> Convert Zero Value
        # isdetect = 0 # 0 : can not detect target drone / 1 : can detect target drone
        # for i in range(self.n_agents):
        #     if (self.angle_list[i][2] == 0):
        #         self.angle_list[i][2] += 1e-5

        #     heading_vector = [math.cos(self.angle_list[i][2]), math.sin(self.angle_list[i][2])]
        #     vector_itoj = self.pos_list[self.start_target_index][:2] - self.pos_list[i][:2]

        #     if (cal_angle_2d_vec(heading_vector, vector_itoj) < self.agent_fov / 2 * d2r):
        #         # Check Obstacle
        #         y1 = (self.pos_list[i][1] - self.pos_list[self.start_target_index][1])/(self.pos_list[i][0] - self.pos_list[self.start_target_index][0]) * (10 - self.pos_list[i][0]) + self.pos_list[i][1]
        #         y2 = (self.pos_list[i][1] - self.pos_list[self.start_target_index][1])/(self.pos_list[i][0] - self.pos_list[self.start_target_index][0]) * (15 - self.pos_list[i][0]) + self.pos_list[i][1]
        #         x1 = (self.pos_list[i][0] - self.pos_list[self.start_target_index][0])/(self.pos_list[i][1] - self.pos_list[self.start_target_index][1]) * (10 - self.pos_list[i][1]) + self.pos_list[i][0]
        #         if not (7.5 < y1 < 12.5 or 7.5 < y2 < 12.5 or 7.5 < x1 < 17.5):
        #             isdetect = 1

        # if (isdetect == 0):
        #     for i in range(self.n_agents):
        #         state_list[i][:7] = [0, ]

        ############################################################################
        # target pos, target vel, target yaw, own pos, own vel, own yaw(cos, sin), own vel cmd, friends pos, friends vel, friends yaw(cos, sin)
        # 2 2 / 2 2 2 3 / 2 2 2 ==> 19
        state_list = [0 for i in range(self.n_agents)]
        for i in range(self.n_agents):
            state_list[i] = ((self.pos_err_jtoi_from_j_frame[self.start_target_index][i] - 10)/10.0).tolist()[:2] + self.vel_err_jtoi_from_j_frame[self.start_target_index][i].tolist()[:2]
            state_list[i] = state_list[i] + ((self.pos_list[i] - 10)/10.0).tolist()[:2] + self.vel_list[i].tolist()[:2] + [math.cos(self.angle_list[i][2]), math.sin(self.angle_list[i][2])] + self.vel_cmd_list[i].tolist()[:2] + [self.vel_cmd_list[i].tolist()[3]]
            for j in range(self.n_agents):
                if j != i:
                    state_list[i] = state_list[i] + ((self.pos_err_jtoi_from_j_frame[j][i] - 10)/10.0).tolist()[:2] + self.vel_err_jtoi_from_j_frame[j][i].tolist()[:2] + [math.cos(self.angle_list[j][2]), math.sin(self.angle_list[j][2])]
            state_list[i] = np.array(state_list[i])

        # Mask Part --> Convert Zero Value
        isdetect = 0 # 0 : can not detect target drone / 1 : can detect target drone
        for i in range(self.n_agents):
            if (self.angle_list[i][2] == 0):
                self.angle_list[i][2] += 1e-5

            heading_vector = [math.cos(self.angle_list[i][2]), math.sin(self.angle_list[i][2])]
            vector_itoj = self.pos_list[self.start_target_index][:2] - self.pos_list[i][:2]

            if (cal_angle_2d_vec(heading_vector, vector_itoj) < self.agent_fov / 2 * d2r):
                # Check Obstacle
                y1 = (self.pos_list[i][1] - self.pos_list[self.start_target_index][1])/(self.pos_list[i][0] - self.pos_list[self.start_target_index][0]) * (10 - self.pos_list[i][0]) + self.pos_list[i][1]
                y2 = (self.pos_list[i][1] - self.pos_list[self.start_target_index][1])/(self.pos_list[i][0] - self.pos_list[self.start_target_index][0]) * (15 - self.pos_list[i][0]) + self.pos_list[i][1]
                x1 = (self.pos_list[i][0] - self.pos_list[self.start_target_index][0])/(self.pos_list[i][1] - self.pos_list[self.start_target_index][1]) * (10 - self.pos_list[i][1]) + self.pos_list[i][0]
                xmin = min(self.pos_list[i][0], self.pos_list[self.start_target_index][0])
                xmax = max(self.pos_list[i][0], self.pos_list[self.start_target_index][0])
                ymin = min(self.pos_list[i][1], self.pos_list[self.start_target_index][1])
                ymax = max(self.pos_list[i][1], self.pos_list[self.start_target_index][1])
                if not ((7.5 < y1 < 12.5 and ymin < y1 < ymax) or (7.5 < y2 < 12.5 and ymin < y2 < ymax) or (7.5 < x1 < 17.5 and xmin < x1 < xmax)):
                    isdetect = 1

        if (isdetect == 0):
            for i in range(self.n_agents):
                state_list[i][:4] = [0, ]

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
        while (np.amax(abs(self.pos_list - past_initial_position_list), axis=0)[2] > 1.0):
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
        while (np.amax(abs(self.pos_list - self.inital_position), axis=0)[2] > 1.0):
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

        for i in range(self.n_agents):
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
            if (time.time() - self.last_detect_time > 3):
                self.detected_drone = [0, 0] # 0 : can not detect target drone / 1 : can detect target drone
                for i in range(self.n_agents):
                    # Check Obstacle
                    y1 = (self.pos_list[i][1] - self.pos_list[self.start_target_index][1])/(self.pos_list[i][0] - self.pos_list[self.start_target_index][0]) * (10 - self.pos_list[i][0]) + self.pos_list[i][1]
                    y2 = (self.pos_list[i][1] - self.pos_list[self.start_target_index][1])/(self.pos_list[i][0] - self.pos_list[self.start_target_index][0]) * (15 - self.pos_list[i][0]) + self.pos_list[i][1]
                    x1 = (self.pos_list[i][0] - self.pos_list[self.start_target_index][0])/(self.pos_list[i][1] - self.pos_list[self.start_target_index][1]) * (10 - self.pos_list[i][1]) + self.pos_list[i][0]
                    xmin = min(self.pos_list[i][0], self.pos_list[self.start_target_index][0])
                    xmax = max(self.pos_list[i][0], self.pos_list[self.start_target_index][0])
                    ymin = min(self.pos_list[i][1], self.pos_list[self.start_target_index][1])
                    ymax = max(self.pos_list[i][1], self.pos_list[self.start_target_index][1])
                    if not ((7.5 < y1 < 12.5 and ymin < y1 < ymax) or (7.5 < y2 < 12.5 and ymin < y2 < ymax) or (7.5 < x1 < 17.5 and xmin < x1 < xmax)):
                        self.detected_drone[i] += 1
                self.last_detect_time = time.time()

            z_vel = satmax(1.5*(self.inital_position[self.start_target_index][2] - self.pos_list[self.start_target_index][2]), 1.0) + 0.3*(0.0 - self.vel_list[self.start_target_index][2])

            # print(self.detected_drone)

            if (sum(self.detected_drone) == 0):
                if (inobstacle(self.pos_list[self.start_target_index][:2], 6.5, 18.5, 6.5, 14.5)):
                    closest_point = find_closest_point(self.pos_list[self.start_target_index][:2], 6.5, 18.5, 6.5, 14.5)
                    heading_vector = (np.array(closest_point) - self.pos_list[self.start_target_index][:2])
                    heading_angle = math.atan(heading_vector[1]/heading_vector[0])
                    self.goal_list[self.start_target_index].data = [8.0] + closest_point + [10, heading_angle] + [1.0, 1.0]
                else:
                    self.goal_list[self.start_target_index].data = [9.0] + [0, 0, z_vel] + [self.angle_list[self.start_target_index][2]] + [0.0, 0.0]
            elif (sum(self.detected_drone) == 1):
                if (self.detected_drone[0] == 1):
                    visible_drone_index = 0
                else:
                    visible_drone_index = 1

                #           |
                #      2    |    1
                # ----------------------
                #      3    |    4
                #           |
                if (self.pos_list[visible_drone_index][0] < 10 and self.pos_list[visible_drone_index][1] < 10): # 3
                    min_point = [20, 20]
                elif (self.pos_list[visible_drone_index][0] < 10): # 2
                    min_point = [20, 0]
                elif (self.pos_list[visible_drone_index][1] > 10): # 1
                    min_point = [0, 0]
                else: # 4
                    min_point = [0, 20]

                # cur_pos, target_pos, xmin, xmax, ymin, ymax
                next_target_pos = avoid_planning(self.pos_list[self.start_target_index][:2], min_point, 6.5, 18.5, 6.5, 14.5)
                heading_vector = (np.array(next_target_pos) - self.pos_list[self.start_target_index][:2])
                heading_angle = math.atan(heading_vector[1]/heading_vector[0])

                next_target_pos = (np.array(next_target_pos) - self.offset_list[self.start_target_index][:2]).tolist()
                self.goal_list[self.start_target_index].data = [8.0] + next_target_pos + [10, heading_angle] + [1.0, 1.0]
            else:
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
                next_target_pos = avoid_planning(self.pos_list[self.start_target_index][:2], min_point, 6.5, 18.5, 6.5, 14.5)
                heading_vector = (np.array(next_target_pos) - self.pos_list[self.start_target_index][:2])
                heading_angle = math.atan(heading_vector[1]/heading_vector[0])

                next_target_pos = (np.array(next_target_pos) - self.offset_list[self.start_target_index][:2]).tolist()
                self.goal_list[self.start_target_index].data = [8.0] + next_target_pos + [10, heading_angle] + [1.0, 1.0]
        elif (self.target_manuver == 3):
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
            next_target_pos = avoid_planning(self.pos_list[self.start_target_index][:2], min_point, 6.5, 18.5, 6.5, 14.5)
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
        past_initial_position_list = []
        for i in range(self.total_drone):
            past_initial_position_list.append([self.pos_list[i][0], self.pos_list[i][1], 10 * (i+2)])


        if (time.time() - self.start_time > self.time_constrain):
            print("Time Constrain")
            if (self.write_command_file == True):
                self.file.close()
                self.file = None

            for i in range(self.total_drone):
                self.goal_list[i].data = [3.0] + past_initial_position_list[i] + [self.start_angle[i]] + [0.0, 1.0]
                self.pub_list[i].publish(self.goal_list[i])
            return True

        distance_target_list = []
        for i in range(self.start_target_index, self.total_drone):
            for j in range(self.n_agents):
                distance_target_list.append(mag_3d_vec(self.pos_err_jtoi_from_j_frame[i][j]))
        if (min(distance_target_list) < self.min_distance):
            self.capture_done = 1
            print("Capture Success")
            if (self.write_command_file == True):
                self.file.close()
                self.file = None

            for i in range(self.total_drone):
                self.goal_list[i].data = [3.0] + past_initial_position_list[i] + [self.start_angle[i]] + [0.0, 1.0]
                self.pub_list[i].publish(self.goal_list[i])
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
            
            for i in range(self.total_drone):
                self.goal_list[i].data = [3.0] + past_initial_position_list[i] + [self.start_angle[i]] + [0.0, 1.0]
                self.pub_list[i].publish(self.goal_list[i])
            return True

        for i in range(self.start_target_index):
            min_cnt = 0
            for j in range(self.n_laser):
                if (self.laser_list[i][j] > 0.9):
                    min_cnt += 1
            if (min_cnt > 10):
                self.collision_done = 1
                print("Collision Warning with Obstacle by laser")
                if (self.write_command_file == True):
                    self.file.close()
                    self.file = None

                for t in range(self.total_drone):
                    self.goal_list[t].data = [3.0] + past_initial_position_list[t] + [self.start_angle[t]] + [0.0, 1.0]
                    self.pub_list[t].publish(self.goal_list[t])
                return True

            if(distance_2d_vec(self.pos_list[i][:2], np.array([10, 10])) < 3.5 or distance_2d_vec(self.pos_list[i][:2], np.array([15, 10])) < 3.5):
                self.collision_done = 1
                print("Collision Warning with Obstacle by distance")
                if (self.write_command_file == True):
                    self.file.close()
                    self.file = None
                
                for t in range(self.total_drone):
                    self.goal_list[t].data = [3.0] + past_initial_position_list[t] + [self.start_angle[t]] + [0.0, 1.0]
                    self.pub_list[t].publish(self.goal_list[t])
                return True

        for i in range(self.total_drone):
            if (self.pos_list[i][0] < -5 or self.pos_list[i][0] > 25 or self.pos_list[i][0] < -5 or self.pos_list[i][1] > 25):
                self.out_field_done = 1
                print("Off field")
                if (self.write_command_file == True):
                    self.file.close()
                    self.file = None

                for t in range(self.total_drone):
                    self.goal_list[t].data = [3.0] + past_initial_position_list[t] + [self.start_angle[t]] + [0.0, 1.0]
                    self.pub_list[t].publish(self.goal_list[t])
                return True

    def get_reward(self):
        # check whether drone close to target
        target_distance_list = []
        for i in range(self.n_agents):
            target_distance_list.append(distance_3d_vec(self.pos_list[self.start_target_index], self.pos_list[i]))

        if (self.reward_type == 1): # Sparse Reward
            reward = np.zeros(self.n_agents)
            for i in range(self.n_agents):
                if (self.capture_done == 1):
                    reward[i] += 1000
                if (self.collision_done == 1):
                    reward[i] += -1000
                if (self.out_field_done == 1):
                    reward[i] += -1000
                reward[i] -= 1
        elif (self.reward_type == 2): # Dense Reward
            distance_reward_list = np.zeros(self.n_agents)
            penalty_reward_list = np.zeros(self.n_agents)

            for target_distance in target_distance_list:
                # Distance
                if (target_distance < self.min_distance):
                    distance_reward = 1000 - (time.time() - self.start_time) * 10
                elif (target_distance < 20):
                    distance_reward = 0.5 - target_distance / 40.0
                else:
                    distance_reward = 0
                distance_reward_list[i] = distance_reward

            for pos in self.pos_list[:self.start_target_index]:
                # Collision and Defense
                if (self.out_field_done):
                    penalty_reward = -1000
                elif (self.collision_done):
                    penalty_reward = -1000
                else:
                    penalty_reward = 0
                penalty_reward_list[i] = penalty_reward

            reward = 1.0 * distance_reward_list + penalty_reward_list
        return reward