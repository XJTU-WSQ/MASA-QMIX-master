import math

import numpy as np


def distance(initial_pos, end_pos, speed):
    return math.sqrt((end_pos[0] - initial_pos[0]) ** 2 + (end_pos[1] - initial_pos[1]) ** 2) / speed


def count_path_on_road(initial_pos, end_pos, speed):
    return (abs(end_pos[0] - initial_pos[0]) + abs(end_pos[1] - initial_pos[1])) / speed


def min_but_zero(state_left_time):
    non_zero_list = []
    for eve in state_left_time:
        if eve != 0:
            non_zero_list.append(eve)
    if len(non_zero_list) != 0:
        return min(non_zero_list)
    else:
        return 0


# 将state_left_time中非0的都减去min_time
def advance_by_min_time(min_time, state_left_time):
    res = []
    for eve in state_left_time:
        if eve != 0:
            res.append(max(eve - min_time, 0))
        else:
            res.append(0)
    return np.array(res)
