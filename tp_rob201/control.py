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
    d_treshold = 30
    d_safe = 100
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

    k_v = 0.003
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

def dwa_control(lidar, current_pose, goal_pose):
    """
    Control using Dynamic Window Approach (DWA) for goal reaching and obstacle avoidance
    [Fox et al., 1997] - Version Corrigée (Anti-Freeze)
    """
    d_goal = np.linalg.norm(goal_pose[:2] - current_pose[:2])

    if d_goal < 5.0:
        return {"forward": 0.0, "rotation": 0.0}

    # 1. Obtenir les obstacles locaux depuis le LiDAR
    ranges = lidar.get_sensor_values()
    angles = lidar.get_ray_angles()
    
    valid = np.logical_and(np.isfinite(ranges), ranges > 1e-6)
    valid = np.logical_and(valid, ranges < 150.0) 
    r = ranges[valid]
    a = angles[valid]
    
    ox = current_pose[0] + r * np.cos(a + current_pose[2])
    oy = current_pose[1] + r * np.sin(a + current_pose[2])
    obstacles = np.column_stack((ox, oy))

    best_cmd = {"forward": 0.0, "rotation": 0.0}
    best_score = -float('inf')

    # 2. Fenêtre Dynamique : On autorise désormais une légère marche arrière (-0.2)
    v_range = np.linspace(-0.2, 1.0, 7)    
    w_range = np.linspace(-1.0, 1.0, 21)  

    dt = 0.2
    predict_time = 2.0
    steps = int(predict_time / dt)
    
    V_SCALE = 15.0 
    W_SCALE = 1.0

    for v in v_range:
        for w in w_range:
            # --- Trajectory Prediction ---
            x, y, theta = current_pose
            v_phys = v * V_SCALE
            w_phys = w * W_SCALE
            
            pts = np.zeros((steps, 2))
            for i in range(steps):
                x += v_phys * np.cos(theta) * dt
                y += v_phys * np.sin(theta) * dt
                theta += w_phys * dt
                pts[i] = [x, y]
                
            final_pose = np.array([x, y, theta])
            
            # --- Clearance Evaluation ---
            min_dist = float('inf')
            if len(obstacles) > 0:
                diff = pts[:, np.newaxis, :] - obstacles[np.newaxis, :, :]
                dists = np.linalg.norm(diff, axis=2)
                min_dist = np.min(dists)
                
            # CORRECTION 1 : On ne rejette la trajectoire que si le robot AVANCE vers l'obstacle.
            # Cela lui permet de pivoter sur place (v=0) ou reculer (v<0) même s'il est très près du mur !
            if min_dist < 15.0 and v > 0.0:  
                continue
                
            # --- Heading Evaluation ---
            dx = goal_pose[0] - final_pose[0]
            dy = goal_pose[1] - final_pose[1]
            angle_to_goal = np.arctan2(dy, dx)
            
            heading_diff = angle_to_goal - final_pose[2]
            heading_diff = np.arctan2(np.sin(heading_diff), np.cos(heading_diff))
            heading_score = (np.pi - abs(heading_diff)) / np.pi
            
            # --- Velocity Evaluation ---
            vel_score = v
            dist_score = min(min_dist, 50.0) / 50.0
            
            # CORRECTION 2 : Rééquilibrage des poids
            # La priorité n°1 est désormais de conserver de la vitesse, le cap devient secondaire.
            alpha = 0.5  # Poids du cap abaissé (avant 2.0)
            beta  = 1.0  # Poids de l'évitement
            gamma = 2.0  # Poids de la vélocité augmenté (avant 0.5)
            
            score = alpha * heading_score + beta * dist_score + gamma * vel_score
            
            # CORRECTION 3 : Pénalité d'arrêt. On déteste l'idée de s'arrêter complètement.
            if v == 0.0 and w == 0.0:
                score -= 1.0
                     
            if score > best_score:
                best_score = score
                best_cmd = {"forward": float(v), "rotation": float(w)}

    # 3. Comportement de récupération profond
    if best_score == -float('inf'):
        # S'il n'y a aucune issue (coincé dans un coin étroit), on force une marche arrière en tournant
        return {"forward": -0.2, "rotation": 1.0}

    return best_cmd

def potential_field_control_2(lidar, current_pose, goal_pose, state):
    """
    Control V2 : Champs de potentiels avec segmentation LIDAR et échappement des minimums locaux.
    """
    # Initialisation du state si vide
    if not state:
        state = {'stuck_counter': 0, 'recovery': False, 'recovery_timer': 0, 'vortex_dir': 1}

    goal_vec = goal_pose[:2] - current_pose[:2]
    d_goal = np.linalg.norm(goal_vec)

    if d_goal < 5.0:
        return {"forward": 0.0, "rotation": 0.0}, state

    #  1. FORCE ATTRACTIVE 
    k_att = 0.05
    d_treshold = 50
    if d_goal > d_treshold:
        att_vec = (k_att / max(d_goal, 1e-6)) * goal_vec 
    else:
        att_vec = (k_att / d_treshold) * goal_vec

    #  2. FORCE RÉPULSIVE AVEC SEGMENTATION 
    ranges = lidar.get_sensor_values()
    angles = lidar.get_ray_angles()

    valid = np.logical_and(np.isfinite(ranges), ranges > 1e-6)
    valid_ranges = ranges[valid]
    valid_angles = angles[valid]

    d_safe = 300.0
    k_rep = 15.0  
    rep_vec = np.zeros(2)

    if len(valid_ranges) > 0:
        # Conversion polaire -> cartésien dans le repère du robot
        xs = valid_ranges * np.cos(valid_angles)
        ys = valid_ranges * np.sin(valid_angles)
        points = np.column_stack((xs, ys))

        # Clustering : séparation si la distance entre 2 rayons consécutifs est > 20 unités
        cluster_threshold = 20.0
        clusters = []
        current_cluster = [0]

        for i in range(1, len(points)):
            if np.linalg.norm(points[i] - points[i-1]) < cluster_threshold:
                current_cluster.append(i)
            else:
                clusters.append(current_cluster)
                current_cluster = [i]
        
        # Gestion du wrap-around (fermeture du 360° LIDAR)
        if len(clusters) > 1 and np.linalg.norm(points[0] - points[-1]) < cluster_threshold:
            clusters[0].extend(clusters[-1])
            clusters.pop()
        else:
            clusters.append(current_cluster)

        # Application de la force pour le point le plus proche de CHAQUE cluster
        for cluster in clusters:
            c_ranges = valid_ranges[cluster]
            c_angles = valid_angles[cluster]
            
            min_idx = np.argmin(c_ranges)
            r_min = c_ranges[min_idx]
            a_min = c_angles[min_idx]

            if r_min < d_safe:
                # Vecteur de répulsion dans le monde absolu
                angle_world = a_min + current_pose[2]
                obs_vec = np.array([np.cos(angle_world), np.sin(angle_world)])
                
                weight = k_rep * (1.0 / r_min - 1.0 / d_safe) / (r_min ** 2)
                rep_vec -= weight * obs_vec * r_min 

    #  3. DÉTECTION ET GESTION DES MINIMUMS LOCAUX 
    field_vec = att_vec + rep_vec
    force_mag = np.linalg.norm(field_vec)

    # Si la force est presque nulle mais qu'on n'est pas à l'objectif = Bloqué
    if force_mag < 0.005 and d_goal > 15.0:
        state['stuck_counter'] += 1
    else:
        state['stuck_counter'] = max(0, state['stuck_counter'] - 1)

    # Déclenchement de la récupération (bloqué depuis ~10 itérations)
    if state['stuck_counter'] > 10:
        state['recovery'] = True
        state['recovery_timer'] = 25  # Durée du mode vortex
        state['stuck_counter'] = 0
        state['vortex_dir'] = np.random.choice([-1, 1]) # Sens d'évitement aléatoire

    # Comportement de récupération (Champ Vortex Perpendiculaire)
    if state['recovery']:
        state['recovery_timer'] -= 1
        if state['recovery_timer'] <= 0:
            state['recovery'] = False
        
        # Vecteur perpendiculaire à l'objectif pour glisser le long de l'obstacle
        perp_vec = np.array([-goal_vec[1], goal_vec[0]])
        perp_vec = perp_vec / np.linalg.norm(perp_vec)
        field_vec = perp_vec * 0.1 * state['vortex_dir'] # Force arbitraire forte

    #  4. TRADUCTION EN COMMANDES MOTRICES 
    if np.linalg.norm(field_vec) < 1e-6:
        field_vec = att_vec

    desired_heading = np.arctan2(field_vec[1], field_vec[0])
    angle_diff = desired_heading - current_pose[2]
    angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

    k_v = 0.003
    k_w = 0.4

    forward = k_v * d_goal * max(0.0, np.cos(angle_diff))
    if np.abs(angle_diff) > np.deg2rad(60):
        forward = 0.0

    rotation = k_w * angle_diff

    command = {
        "forward": float(np.clip(forward, 0.0, 1.0)),
        "rotation": float(np.clip(rotation, -1.0, 1.0)),
    }

    return command, state