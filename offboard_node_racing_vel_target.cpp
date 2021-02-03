
#include<iostream>

#include <ros/ros.h>
#include <std_msgs/String.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include <mavros_msgs/AttitudeTarget.h>

#include <std_msgs/UInt8.h>
#include <std_msgs/UInt8MultiArray.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Float64MultiArray.h>
//#include <tf2/LinearMath/Quaternion.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>

#include <nav_msgs/Odometry.h>

#include "SubtUtil_racing.h"

ros::Time landing_request;
ros::Time takeoff_request;
ros::Time hovering_request;

float ucmd, vcmd, wcmd, rcmd;
float ucmd_pre, vcmd_pre, wcmd_pre, rcmd_pre;
float ucmd_LPF, vcmd_LPF, wcmd_LPF, rcmd_LPF;

mavros_msgs::State             g_current_state;
geometry_msgs::TwistStamped    velcmd;
geometry_msgs::PoseStamped     Local;
geometry_msgs::TwistStamped    Localvel;
nav_msgs::Odometry             Subt_Odom, Odom;

std_msgs::Float32MultiArray    body_vel;
std_msgs::Float32MultiArray    hover_pt;
std_msgs::UInt8                mission;
std_msgs::UInt8                MotorAct_msg;
std_msgs::UInt8                FlightMode_msg;
tf::Quaternion                 quat;

std_msgs::Float32MultiArray    tele_flag;
geometry_msgs::Twist           tele_cmd;
std_msgs::Float32MultiArray    GoalAction;
std_msgs::Float32MultiArray    Target_Sub;

float   cmd_x = 0.0;
float   cmd_y = 0.0;
float   cmd_z = 0.0;
float   cmd_r = 0.0;

float   Cur_Pos_m[3];
float   Cur_Vel_mps[3];
float   Cur_Att_rad[3];
float   Cur_Att[3];
float   cmd_rpy[3];
float   height_m = 0.0;

float   pos_x = 0.0;
float   pos_y = 0.0;
float   pos_z = 0.0;

float   cur_x = 0.0;
float   cur_y = 0.0;
float   cur_z = 0.0;
float   angle_err = 0.0;

float   takeoff_x = 0.0;
float   takeoff_y = 0.0;
float   init_heading = 0.0;

float   emer[2];
float   hover[3];
float   goal_dist = 10.0;
float   goal_heading = 0.0;
float   hover_heading = 0.0;

float   crusing_height = 0.0;
float   takeoff_height = 0.0;
float   sp_pos_cmd[3];

int     flag_takeoff = 0;
int     flag_landing = 0;
int     flag_goal = 0;
int     flag_turning = 0;
int     flag_hovering = 0;

int     turning_dir = 0;
int     WP_index = 0;
uint8_t   MotorAct = 0;
uint8_t   FlightMode = 0;
int       flag_armed = 0;
int       count = 0;
int       goal_service = 0;

float    goal[4] = {0, };
float    goal_velx;
float    goal_velz;

float    target_err[3];

float    WayPoint_X[4];
float    WayPoint_Y[4];
float    WayPoint_R[4];
float    WayPoint_U[4];
float    WayPoint_V[4];

// Waypoint Visualization
#define eta 1
#define rad 8.0
#define mag 0.0
#define alt 3.0
#define period 5.0

#define fq 20.0
#define vel 1.5
#define center_x 5.0
#define center_y -4.0

double duration_to_float(const ros::Duration t)
{
    int sec = t.sec;
    int nsec = t.nsec;

    return (double)sec + (double)nsec * 0.000000001;
}


void callback_cmd_flag(const std_msgs::Float32MultiArray::ConstPtr& msg)
{
    tele_flag = *msg;
}

void callback_cmd_vel(const geometry_msgs::Twist::ConstPtr& msg)
{
    tele_cmd = *msg;
}

void callback_goal(const std_msgs::Float32MultiArray::ConstPtr& msg)
{
    GoalAction = *msg;

    goal_service = GoalAction.data[0];

    goal[0] = GoalAction.data[1];
    goal[1] = GoalAction.data[2];
    goal[2] = GoalAction.data[3];
    goal[3] = GoalAction.data[4];

    goal_velx = GoalAction.data[5];
    goal_velz = GoalAction.data[6];
}

void callback_target(const std_msgs::Float32MultiArray::ConstPtr& msg)
{
    Target_Sub = *msg;

    target_err[0] = Target_Sub.data[0];
    target_err[1] = Target_Sub.data[1];
    target_err[2] = Target_Sub.data[2];
}

void callback_state(const mavros_msgs::State::ConstPtr& msg)
{
    g_current_state = *msg;

    std::cout << "\n[USRG] state_cb(), -----------";
    std::cout << "\n          g_current_state.connected = " << ((g_current_state.connected) ? "OK!" : "Not yet!");
    std::cout << "\n          g_current_state.armed     = " << ((g_current_state.armed ) ? "OK!" : "Not yet!");
    std::cout << "\n          g_current_state.guided    = " << ((g_current_state.guided) ? "OK!" : "Not yet!");
    std::cout << "\n          g_current_state.mode      = " << g_current_state.mode;
    std::cout << "\n          Cur   X Y Z r             = " << Cur_Pos_m[0] << ", "<< Cur_Pos_m[1] << ", "<< Cur_Pos_m[2] << ", "<<  Cur_Att_rad[2]*R2D;
    std::cout << "\n          Target Err X Y Z          = " << target_err[0] << ", " << target_err[1] << ", " << target_err[2];
    std::cout << "\n          Drone Frame velocity      = " << goal[0] << ", "<<goal[1] << ", "<<goal[2] << ", "<<goal[3];
    std::cout << "\n          Sim Frame velocity        = " << velcmd.twist.linear.x << ", "<<velcmd.twist.linear.y << ", "<<velcmd.twist.linear.z <<  ", "<<velcmd.twist.angular.z;
    std::cout << "\n          goal service              = " << goal_service;
    std::cout << "\n[USRG] ------------------------\n";
}

std_msgs::Float32MultiArray pathdata;
void callback_astar_path(const std_msgs::Float32MultiArray::ConstPtr& msg_input)
{
    pathdata = *msg_input;

    path.x = pathdata.data[0];
    path.y = pathdata.data[1];
    path.psi = pathdata.data[2];
}

void callback_local_pos(const geometry_msgs::PoseStamped::ConstPtr& msg_input)
{
    Local = *msg_input;

    Cur_Pos_m[0] = Local.pose.position.x;
    Cur_Pos_m[1] = Local.pose.position.y;
    Cur_Pos_m[2] = Local.pose.position.z;

    q[0] = Local.pose.orientation.x;
    q[1] = Local.pose.orientation.y;
    q[2] = Local.pose.orientation.z;
    q[3] = Local.pose.orientation.w;
    QuaterniontoEuler(Cur_Att_rad[0], Cur_Att_rad[1], Cur_Att_rad[2]);

    quat[0] = q[0];
    quat[1] = q[1];
    quat[2] = q[2];
    quat[3] = q[3];

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    transform.setOrigin(tf::Vector3(Cur_Pos_m[0], Cur_Pos_m[1], Cur_Pos_m[2]));
    transform.setRotation(quat);
    br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "map", "racing"));
}


void callback_local_vel(const geometry_msgs::TwistStamped::ConstPtr& msg_input)
{
    Localvel = *msg_input;

    Cur_Vel_mps[0] = Localvel.twist.linear.x;
    Cur_Vel_mps[1] = Localvel.twist.linear.y;
    Cur_Vel_mps[2] = Localvel.twist.linear.z;
}

void Mission_Update(void);
void Auto_Takeoff(void);
void Auto_Landing(void);
void WP_Flight(void);
void Path_Flight(void);
void Hovering(void);
void RL_action(void);
void Evation(void);
void Relative_Vel(void);

void sim2drone(float sim_pos[3], float euler_matrix[3], float ret_pos[3]);
void drone2sim(float drone_pos[3], float euler_matrix[3], float ret_pos[3]);
void multiply(float mat1[3][3], float mat2[3][3], float res[3][3]);

int main(int argc, char **argv)
{
    ros::init(argc, argv, "offboard_node_racing_vel_target");
    ros::NodeHandle nh_sub;
    ros::NodeHandle nh_pub;

    // Subscribe Topic
    ros::Subscriber state_sub     = nh_sub.subscribe ("/UAV_2/mavros/state" , 2,                         &callback_state);
    ros::Subscriber local_pos_sub = nh_sub.subscribe ("/UAV_2/mavros/local_position/pose", 2,            &callback_local_pos);
    ros::Subscriber local_vel_sub = nh_sub.subscribe ("/UAV_2/mavros/local_position/velocity_local", 2,  &callback_local_vel);
    ros::Subscriber cmd_sub       = nh_sub.subscribe ("/UAV_2/mavros_comm_node/tele_key/cmd_vel", 2,     &callback_cmd_vel);
    ros::Subscriber flag_sub      = nh_sub.subscribe ("/UAV_2/mavros_comm_node/tele_key/flag", 2,        &callback_cmd_flag);
    ros::Subscriber target_sub    = nh_sub.subscribe ("/UAV_2/targret_pos", 2,                           &callback_target);

    ros::Subscriber astar_sub     = nh_sub.subscribe ("/UAV_2/astar_path_info", 2,                       &callback_astar_path);
    ros::Subscriber goal_sub      = nh_sub.subscribe ("/UAV_2/GoalAction", 2,                            &callback_goal);

    // Publish Topic
    ros::Publisher  local_vel_pub = nh_pub.advertise<geometry_msgs::TwistStamped>("/UAV_2/mavros/setpoint_velocity/cmd_vel", 2);

    ros::ServiceClient  arming_client    = nh_pub.serviceClient<mavros_msgs::CommandBool> ("/UAV_2/mavros/cmd/arming");
    ros::ServiceClient  set_mode_client  = nh_pub.serviceClient<mavros_msgs::SetMode>     ("/UAV_2/mavros/set_mode");

    //the setpoint publishing rate MUST be faster than 2Hz
    ros::Rate rate(30.0);

    // wait for FCU connection
    while(ros::ok() && g_current_state.connected){
        ros::spinOnce();
        rate.sleep();
    }

    ucmd = 0.0;
    vcmd = 0.0;
    wcmd = 0.0;
    rcmd = 0.0;

    cmd_x = 0.0;
    cmd_y = 0.0;
    cmd_z = 0.0;
    cmd_r = 0.0;

    velcmd.twist.linear.x = 0.0;
    velcmd.twist.linear.y = 0.0;
    velcmd.twist.linear.z = 0.0;
    velcmd.twist.angular.z = 0.0;

    tele_flag.data.resize(5);
    body_vel.data.resize(4);
    hover_pt.data.resize(4);

    //send a few setpoints before starting
    for(int i = 100; ros::ok() && i > 0; --i)
    {
        local_vel_pub.publish(velcmd);
        ros::spinOnce();
        rate.sleep();
    }

    mavros_msgs::SetMode offb_set_mode;
    offb_set_mode.request.custom_mode = "OFFBOARD";

    mavros_msgs::CommandBool arm_cmd;
    mavros_msgs::CommandBool disarm_cmd;
    arm_cmd.request.value = true;
    disarm_cmd.request.value = false;

    ros::Time last_request = ros::Time::now();
    printf("offboard start\n");

    // tele_flag.data[0] flag for arming    -- "t"
    // tele_flag.data[1] flag for auto      -- "g"

    mission.data = 0;  // [None]
    tele_flag.data[1] = 1; //?

    ros::Time prev_time = ros::Time::now();
    ros::Time cur_time = ros::Time::now();

    while(ros::ok())
    {
        // if( g_current_state.mode != "OFFBOARD" && ((ros::Time::now() - last_request) > ros::Duration(1.0)))
        // {

        //     if( set_mode_client.call(offb_set_mode) &&
        //         offb_set_mode.response.mode_sent){
        //         ROS_INFO("Offboard enabled");
        //     }
        //     last_request = ros::Time::now();

        // }
        // else
        // {
        //     if (tele_flag.data[0] == 1)
        //     {
        //         if(flag_armed == 0)
        //         {
        //             if((!g_current_state.armed) && (ros::Time::now() - last_request > ros::Duration(1.0)))
        //             {
        //                 if( arming_client.call(arm_cmd) && arm_cmd.response.success)
        //                 {
        //                     flag_armed = 1;
        //                     ROS_INFO("Vehicle armed");
        //                     MotorAct = 1;
        //                 }
        //                 last_request = ros::Time::now();
        //             }
        //         }
        //     }
        //     else
        //     {
        //         if ((flag_armed == 1))
        //         {
        //             flag_armed = 0;
        //             ROS_INFO("Vehicle Disarmed");
        //             arming_client.call(disarm_cmd);
        //             MotorAct = 0;
        //             mission.data = 0;  // [None]
        //             last_request = ros::Time::now();
        //             takeoff_request = ros::Time::now();
        //         }
        //     }
        // }

        if( g_current_state.mode == "OFFBOARD")
        {
            Mission_Update();
        }
        else
        {
            cmd_x = 0.0;
            cmd_y = 0.0;
            cmd_z = 0.0;
            cmd_r = 0.0;

            velcmd.twist.linear.x = 0.0;
            velcmd.twist.linear.y = 0.0;
            velcmd.twist.linear.z = 0.0;
            velcmd.twist.angular.z = 0.0;

            takeoff_x = Cur_Pos_m[0];
            takeoff_y = Cur_Pos_m[1];
            init_heading = Cur_Att_rad[2];
            goal_heading = Cur_Att_rad[2];

        }

        if(tele_flag.data[1] == 0)
        {
            // manual control mode
            ucmd =   (tele_cmd.linear.x)*cos(Cur_Att_rad[2]) + (tele_cmd.linear.y)*sin(Cur_Att_rad[2]);
            vcmd =   (tele_cmd.linear.x)*sin(Cur_Att_rad[2]) - (tele_cmd.linear.y)*cos(Cur_Att_rad[2]);
            wcmd =   tele_cmd.linear.z;
            rcmd =   tele_cmd.angular.z;
        }
        else
        {
            // auto control mode
            ucmd =   cmd_x + (tele_cmd.linear.x)*cos(Cur_Att_rad[2]) + (tele_cmd.linear.y)*sin(Cur_Att_rad[2]);
            vcmd =   cmd_y + (tele_cmd.linear.x)*sin(Cur_Att_rad[2]) - (tele_cmd.linear.y)*cos(Cur_Att_rad[2]);
            wcmd =   cmd_z + tele_cmd.linear.z;
            rcmd =   cmd_r + tele_cmd.angular.z;
        }

        velcmd.twist.linear.x = ucmd;
        velcmd.twist.linear.y = vcmd;
        velcmd.twist.linear.z = wcmd;
        velcmd.twist.angular.z = -rcmd;
        local_vel_pub.publish(velcmd);

        cur_time = ros::Time::now();
        // printf("Check Hz : %.3f\n", 1/duration_to_float(cur_time - prev_time));
        prev_time = cur_time;

        count = count + 1;
        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}

void Mission_Update(void)
{

    switch (goal_service)
    {
    // Basic Mission
        case 1:
            Auto_Takeoff();
            break;

        case 2:
            Auto_Landing();
            break;

        case 3:
            WP_Flight();
            break;

        case 4:
            Path_Flight();
            break;

        case 5:
            Hovering();
            break;

        case 7:
            RL_action();
            break;

        case 8:
            Evation();
            break;

        case 9:
            Relative_Vel();
            break;
        
        default:
            cmd_x = 0.0;
            cmd_y = 0.0;
            cmd_z = 0.0;
            cmd_r = 0.0;

            velcmd.twist.linear.x = 0.0;
            velcmd.twist.linear.y = 0.0;
            velcmd.twist.linear.z = 0.0;
            velcmd.twist.angular.z = 0.0;

            takeoff_x = Cur_Pos_m[0];
            takeoff_y = Cur_Pos_m[1];
            init_heading = Cur_Att_rad[2];
            goal_heading = Cur_Att_rad[2];

            goal_dist = 10.0;

            break;
    }
}


void Auto_Takeoff(void)
{
    cmd_x = satmax(Kpx*(takeoff_x - Cur_Pos_m[0]), goal_velx);
    cmd_y = satmax(Kpx*(takeoff_y - Cur_Pos_m[1]), goal_velx);
    cmd_z = satmax(Kpz*(goal[2] - Cur_Pos_m[2]), goal_velz);

    angle_err = GetNED_angle_err(init_heading, Cur_Att_rad[2]);
    cmd_r = -satmax(Kr*angle_err, R_MAX);
    mission.data = 1;  // [takeoff]

    if (fabs(goal[2] - Cur_Pos_m[2]) < 0.1)
    {
         goal_service = 5;
         hover[0] = Cur_Pos_m[0];
         hover[1] = Cur_Pos_m[1];
         hover[2] = goal[2];
         hover_heading = Cur_Att_rad[2];
    }
}


void Auto_Landing(void)
{
    cmd_x = satmax(Kpx*(goal[0] - Cur_Pos_m[0]),goal_velx);
    cmd_y = satmax(Kpx*(goal[1] - Cur_Pos_m[1]),goal_velx);
    cmd_z = goal_velz;

    angle_err = GetNED_angle_err(goal[3], Cur_Att_rad[2]);
    cmd_r = -satmax(Kr*angle_err, R_MAX);

    if (flag_landing != 1)
    {
        landing_request = ros::Time::now();
    }

    if (Cur_Pos_m[2] < 0.2)
    {
        cmd_z = -0.5;
        flag_landing = 1;
        if (ros::Time::now() - landing_request > ros::Duration(1.0))
        {
            flag_takeoff = 0;
            takeoff_x = Cur_Pos_m[0];
            takeoff_y = Cur_Pos_m[1];
            flag_landing = 0;
            goal_service = 0;
            takeoff_request = ros::Time::now();
        }
    }
}

void WP_Flight(void)
{
    cmd_x = satmax(Kpx*(goal[0] - Cur_Pos_m[0]),goal_velx*2);
    cmd_y = satmax(Kpx*(goal[1] - Cur_Pos_m[1]),goal_velx*2);
    cmd_z = satmax(Kpz*(goal[2] - Cur_Pos_m[2]),goal_velz) + Kdz*(0.0 - Cur_Vel_mps[2]);

    angle_err = GetNED_angle_err(goal[3], Cur_Att_rad[2]);
    cmd_r = -satmax(Kr*angle_err, R_MAX);
    mission.data = 3;  // [fly_to]

    hover[0] = Cur_Pos_m[0];
    hover[1] = Cur_Pos_m[1];
    hover[2] = goal[2];
    hover_heading = Cur_Att_rad[2];
    goal_heading = Cur_Att_rad[2];

}

void Path_Flight(void)
{
    cmd_x = goal_velx * cos(path.psi);
    cmd_y = goal_velx * sin(path.psi);
    cmd_z = satmax(Kpz*(goal[2] - Cur_Pos_m[2]), goal_velz);

    angle_err = GetNED_angle_err(goal[3], Cur_Att_rad[2]);
    cmd_r = -satmax(Kr*angle_err, R_MAX);

    hover[0] = Cur_Pos_m[0];
    hover[1] = Cur_Pos_m[1];
    hover[2] = goal[2];
    hover_heading = Cur_Att_rad[2];
    goal_heading = Cur_Att_rad[2];
}

void Hovering(void)
{
    cmd_x = satmax(Kpx*(hover[0] - Cur_Pos_m[0]),goal_velx);
    cmd_y = satmax(Kpx*(hover[1] - Cur_Pos_m[1]),goal_velx);
    cmd_z = satmax(Kpz*(goal[2] - Cur_Pos_m[2]),goal_velz);

    angle_err = GetNED_angle_err(goal[3], Cur_Att_rad[2]);
    cmd_r = -satmax(Kr*angle_err, R_MAX);
    goal_heading = Cur_Att_rad[2];
}

void RL_action(void)
{
    float drone_goal[3] = {goal[0], goal[1], goal[2]};
    float angle[3] = {Cur_Att_rad[0], Cur_Att_rad[1], Cur_Att_rad[2]};
    
    float sim_goal[3] = {0, };
    drone2sim(drone_goal, angle, sim_goal);

    cmd_x = sim_goal[0];
    cmd_y = sim_goal[1];
    cmd_z = sim_goal[2];
    cmd_r = goal[3];
}

void Evation(void)
{
    cmd_x = satmax(Kpx*(goal[0] - Cur_Pos_m[0]),goal_velx);
    cmd_y = satmax(Kpx*(goal[1] - Cur_Pos_m[1]),goal_velx);
    cmd_z = satmax(Kpz*(goal[2] - Cur_Pos_m[2]),goal_velz) + Kdz*(0.0 - Cur_Vel_mps[2]);

    angle_err = GetNED_angle_err(goal[3], Cur_Att_rad[2]);
    cmd_r = -satmax(Kr*angle_err, R_MAX);
}

void Relative_Vel(void)
{
    cmd_x = goal[0];
    cmd_y = goal[1];
    cmd_z = goal[2];

    angle_err = GetNED_angle_err(goal[3], Cur_Att_rad[2]);
    cmd_r = -satmax(Kr*angle_err, R_MAX);
}


void sim2drone(float sim_pos[3], float euler_matrix[3], float ret_pos[3])
{
    float sr = sin(euler_matrix[0]);
    float cr = cos(euler_matrix[0]);
    float sp = sin(euler_matrix[1]);
    float cp = cos(euler_matrix[1]);
    float sy = sin(euler_matrix[2]);
    float cy = cos(euler_matrix[2]);

    float x_matrix[3][3] = {
        1,   0,   0,
        0,  cr,  sr,
        0, -sr,  cr
    };

    float y_matrix[3][3] = {
       cp,   0, -sp,
        0,   1,   0,
       sp,   0,  cp
    };

    float z_matrix[3][3] = {
        cy,  sy,   0,
       -sy,  cy,   0,
         0,   0,   1
    };

    float temp_matrix[3][3] = {0, };
    float ret_matrix[3][3] = {0, };

    multiply(x_matrix, y_matrix, temp_matrix);
    multiply(temp_matrix, z_matrix, ret_matrix);

    static int i, j;
    for (i = 0; i < 3; i++)
    {
        for (j = 0; j < 3; j++)
        {
            ret_pos[i] += ret_matrix[i][j] * sim_pos[j];
        }
    }
}

void drone2sim(float drone_pos[3], float euler_matrix[3], float ret_pos[3])
{
    float sr = sin(euler_matrix[0]);
    float cr = cos(euler_matrix[0]);
    float sp = sin(euler_matrix[1]);
    float cp = cos(euler_matrix[1]);
    float sy = sin(euler_matrix[2]);
    float cy = cos(euler_matrix[2]);

    float x_matrix[3][3] = {
        1,   0,   0,
        0,  cr,  -sr,
        0,  sr,   cr
    };

    float y_matrix[3][3] = {
       cp,   0,  sp,
        0,   1,   0,
      -sp,   0,  cp
    };

    float z_matrix[3][3] = {
        cy, -sy,   0,
        sy,  cy,   0,
         0,   0,   1
    };

    float temp_matrix[3][3] = {0, };
    float ret_matrix[3][3] = {0, };

    multiply(z_matrix, y_matrix, temp_matrix);
    multiply(temp_matrix, x_matrix, ret_matrix);

    static int i, j;
    for (i = 0; i < 3; i++)
    {
        for (j = 0; j < 3; j++)
        {
            ret_pos[i] += ret_matrix[i][j] * drone_pos[j];
        }
    }
}

void multiply(float mat1[3][3], float mat2[3][3], float res[3][3])
{
    static int i, j, k;
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            res[i][j] = 0;
            for (k = 0; k < 3; k++)
                res[i][j] += mat1[i][k] * mat2[k][j];
        }
    }
}