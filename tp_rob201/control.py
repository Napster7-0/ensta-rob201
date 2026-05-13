""" A set of robotics control functions """

import random
import numpy as np


def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    # TODO for TP1

    laser_dist = lidar.get_sensor_values()


    speed = 1.0
    rotation_speed = 0.0

    if (laser_dist.shape[0] > 0 and np.min(laser_dist[175:195]) < 100):
        speed = 0
        rotation_speed = np.where(laser_dist == np.max(laser_dist[180:190]))[0][0] / 360
    
    # Clip commands to reasonable ranges and ensure Python int type
    speed_clipped = float(np.clip(speed, -1, 1).item())
    rot_clipped = float(np.clip(rotation_speed, -1, 1).item())
    
    command = {"forward": speed_clipped,
               "rotation": rot_clipped}

    return command


def potential_field_control(lidar, current_pose, goal_pose):
    """
    Control using potential field for goal reaching and obstacle avoidance
    lidar : placebot object with lidar data
    current_pose : [x, y, theta] nparray, current pose in odom or world frame
    goal_pose : [x, y, theta] nparray, target pose in odom or world frame
    Notes: As lidar and odom are local only data, goal and gradient will be defined either in
    robot (x,y) frame (centered on robot, x forward, y on left) or in odom (centered / aligned
    on initial pose, x forward, y on left)
    """
    goal_vec = goal_pose[:2] - current_pose[:2]
    d_goal = np.linalg.norm(goal_vec)

    if d_goal < 5.0:
        return {"forward": 0.0, "rotation": 0.0}

    k_att = 0.05
    d_treshold = 50
    d_safe = 300
    k_rep = 1
    if d_goal > d_treshold:
        att_vec = (k_att / max(d_goal, 1e-6)) * goal_vec 
    else :
        att_vec = (k_att/d_treshold) * goal_vec

    ranges = lidar.get_sensor_values()
    angles = lidar.get_ray_angles()

    

    valid = np.logical_and(np.isfinite(ranges), ranges > 1e-6)
    close = np.logical_and(valid, ranges < d_safe)

    rep_vec = np.zeros(2)
    if np.any(close):
        r = ranges[close]
        a = angles[close] + current_pose[2]

        # Vector from robot to obstacle in world frame
        obs_vectors = np.column_stack((r * np.cos(a), r * np.sin(a)))
        weights = k_rep * (1.0 / r - 1.0 / d_safe) / (r ** 2)

        # Repulsion points away from obstacles
        rep_vec = -np.sum(obs_vectors * weights[:, None], axis=0)

    field_vec = att_vec + rep_vec
    if np.linalg.norm(field_vec) < 1e-6:
        field_vec = att_vec

    desired_heading = np.arctan2(field_vec[1], field_vec[0])
    angle_diff = desired_heading - current_pose[2]
    angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

    k_v = 0.006
    k_w = 0.3

    forward = k_v * d_goal * max(0.0, np.cos(angle_diff))
    if np.abs(angle_diff) > np.deg2rad(60):
        forward = 0.0

    rotation = k_w * angle_diff

    command = {
        "forward": float(np.clip(forward, 0.0, 1.0)),
        "rotation": float(np.clip(rotation, -1.0, 1.0)),
    }

    return command
