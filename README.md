# GAZEBO_RL
GAZEBO_RL

## Current Setting
1. Callback Hz : 120Hz
2. Offboard Hz : 500Hz
3. RL Hz : 150Hz

## Env
1. ROS melodic
2. Gazebo
3. Python 2.7
4. Pytorch 1.5.1
5. Cuda 10.0
6. Cudnn 7.5.5

## HOW TO USE
1. roslaunch px4 mavros_sitl_posix.launch
2. Open Qground
3. Hz Setting, First check Callback Hz
> If your Callback Hz below 100, then You have to type "mavros stream -r 200 -s LOCAL_POSITION_NED -u {port_num}" in roslaunch px4 terminal
4. rosrun offboard_node offboard_node_racing_rate
5. rosrun rl ppo_lstm_hovering_v2.py
