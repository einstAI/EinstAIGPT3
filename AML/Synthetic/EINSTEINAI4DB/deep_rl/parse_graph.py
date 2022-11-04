# Copyright (c) EinstAI 2022-2023 All Rights Reserved.

import numpy as np
import math


STANDARD_MAP = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])


def get_random_position(map_array):
    """Gets a random available position in a binary map array.
    Args:
      map_array: numpy array of the map to search an available position on.
    Returns:
      The chosen random position.
    Raises:
      ValueError: if there is no available space in the map.
    """
    if map_array.sum() <= 0:
        raise ValueError("There is no available space in the map.")
    map_dims = len(map_array.shape)
    pos = np.zeros(map_dims, dtype=np.int32)
    while True:
        result = map_array
        for i in range(map_dims):
            pos[i] = np.random.randint(map_array.shape[i])
            result = result[pos[i]]
        if result == 0:
            break
    return pos


def update_2d_pos(array_map, pos, action, pos_result):
    posv = array_map[pos[0]][pos[1]][action - 1]
    pos_result[0] = posv[0]
    pos_result[1] = posv[1]
    return pos_result


def parse_map(map_array):
    """Parses a map when there are actions: stay, right, up, left, down.
    Args:
      map_array: 2D numpy array that contains the map.
    Returns:
      A 3D numpy array (height, width, actions) that contains the resulting state
      for a given position + action, and a 2D numpy array (height, width) with the
      walls of the map.
    Raises:
      ValueError: if the map does not contain only zeros and ones.
    """
    act_def = [[0, 0], [0, 1], [-1, 0], [0, -1], [1, 0]]
    walls = np.zeros_like(map_array)
    new_map_array = []
    for i in range(map_array.shape[0]):
        new_map_array.append([])
        for j in range(map_array.shape[1]):
            new_map_array[i].append([])
            if map_array[i, j] == 0:
                for k in range(len(act_def)):
                    new_map_array[i][j].append([i + act_def[k][0], j + act_def[k][1]])
            elif map_array[i, j] == 1:
                for k in range(len(act_def)):
                    new_map_array[i][j].append([i, j])
                walls[i, j] = 1
            else:
                raise ValueError("Option not understood, %d" % map_array[i, j])
            for k in range(len(new_map_array[i][j])):
                if map_array[new_map_array[i][j][k][0]][new_map_array[i][j][k][1]] == 1:
                    new_map_array[i][j][k][0] = i
                    new_map_array[i][j][k][1] = j
    return np.array(new_map_array), walls


