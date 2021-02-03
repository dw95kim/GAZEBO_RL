import math
import numpy as np
import datetime

# Env Part
d2r = math.pi/180
r2d = 180/math.pi
eps = 0.00000001

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

# check whether the drone is in obstacle
# true : in obstacle
# false : otherwise
def inobstacle(pos, xmin, xmax, ymin, ymax):
    if (xmin < pos[0] < xmax and ymin < pos[1] < ymax):
        return 1
    else:
        return 0

def find_closest_point(pos, xmin, xmax, ymin, ymax):
    distance1 = distance_2d_vec(pos, [xmin, ymin])
    distance2 = distance_2d_vec(pos, [xmin, ymax])
    distance3 = distance_2d_vec(pos, [xmax, ymin])
    distance4 = distance_2d_vec(pos, [xmax, ymax])

    distance_list = [distance1, distance2, distance3, distance4]
    min_index = distance_list.index(min(distance_list))

    if (min_index == 0):
        return [xmin, ymin]
    elif (min_index == 1):
        return [xmin, ymax]
    elif (min_index == 2):
        return [xmax, ymin]
    else:
        return [xmax, ymax]


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
            temp1 = distance_2d_vec(cur_pos, [xmax, ymax]) + distance_2d_vec([xmax, ymax], target_pos)
            temp2 = distance_2d_vec(cur_pos, [xmin, ymin]) + distance_2d_vec([xmin, ymin], target_pos)
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
            temp1 = distance_2d_vec(cur_pos, [xmax, ymin]) + distance_2d_vec([xmax, ymin], target_pos)
            temp2 = distance_2d_vec(cur_pos, [xmin, ymax]) + distance_2d_vec([xmin, ymax], target_pos)
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
            temp1 = distance_2d_vec(cur_pos, [xmin, ymax]) + distance_2d_vec([xmin, ymax], target_pos)
            temp2 = distance_2d_vec(cur_pos, [xmax, ymin]) + distance_2d_vec([xmax, ymin], target_pos)
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
            temp1 = distance_2d_vec(cur_pos, [xmax, ymax]) + distance_2d_vec([xmax, ymax], target_pos)
            temp2 = distance_2d_vec(cur_pos, [xmin, ymin]) + distance_2d_vec([xmin, ymin], target_pos)
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
