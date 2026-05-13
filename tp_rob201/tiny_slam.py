""" A simple robotics navigation code including SLAM, exploration, planning"""

import cv2
import numpy as np
from occupancy_grid import OccupancyGrid


class TinySlam:
    """Simple occupancy grid SLAM"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid

        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])


    def score(self, lidar, pose):
        """
        Computes the sum of log probabilities of laser end points in the map
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, position of the robot to evaluate, in world coordinates
        """
        # TODO for TP4
        # les coordonnées des points d'impact lidar dans le repère absolu
        lidar_values = lidar.get_sensor_values()
        lidar_angles = lidar.get_ray_angles()

        #selection des points qui rencontre un impact
        indices = np.where(lidar_values < 500)
        sub_values = lidar_values[indices]
        sub_angles = lidar_angles[indices]

        #conversion dans le repere absolu
        l_x_abs = pose[0] + sub_values * np.cos(sub_angles + pose[2])
        l_y_abs = pose[1] + sub_values * np.sin(sub_angles + pose[2])

        #garder les points qui sont sur la carte
        l_x_map, l_y_map = self.grid.conv_world_to_map(l_x_abs,l_y_abs)
        mask = np.logical_and(
            np.logical_and(l_x_map >= 0, l_x_map < self.grid.x_max_map),
            np.logical_and(l_y_map >= 0, l_y_map < self.grid.y_max_map),
        )
        x = l_x_map[mask]
        y = l_y_map[mask]

        score = self.grid.occupancy_map[x,y].sum()

        return score

    def get_corrected_pose(self, odom_pose, odom_pose_ref=None):
        """
        Compute corrected pose in map frame from raw odom pose + odom frame pose,
        either given as second param or using the ref from the object
        odom : raw odometry position
        odom_pose_ref : optional, origin of the odom frame if given,
                        use self.odom_pose_ref if not given
        """
        # TODO for TP4

        if odom_pose_ref is None:
            odom_pose_ref = self.odom_pose_ref

        d     = np.linalg.norm(odom_pose[0:2])
        alpha = np.arctan2(odom_pose[1], odom_pose[0])

        cor_x = odom_pose_ref[0] + d * np.cos(odom_pose_ref[2] + alpha)
        cor_y = odom_pose_ref[1] + d * np.sin(odom_pose_ref[2] + alpha)
        cor_θ = odom_pose_ref[2] + odom_pose[2]
        return [cor_x, cor_y, cor_θ]


    def localise(self, lidar, raw_odom_pose):
        """
        Compute the robot position wrt the map, and updates the odometry reference
        lidar : placebot object with lidar data
        odom : [x, y, theta] nparray, raw odometry position
        """
        # TODO for TP4

        # 1. Score de la pose courante (référence actuelle)
        cor_pose0   = self.get_corrected_pose(raw_odom_pose)   # utilise self.odom_pose_ref
        best_score  = self.score(lidar, cor_pose0)
        best_ref    = np.copy(self.odom_pose_ref)                  # copie 3D 

        # 2. Hyper-paramètres
        sigma_xy    = 5.0   # unités-monde 
        sigma_theta = 0.05  # radians (~3°)
        N_max       = 1000    # tirages sans amélioration avant arrêt
        N           = 0

        while N < N_max:
        # 3. Tirer un offset 3D (x, y, θ)
            offset = np.random.normal(scale=[sigma_xy, sigma_xy, sigma_theta], size=3)
            candidate = best_ref + offset                     # taille 3 !

        # 4. Calculer le score avec la pose candidate
            cor_pose = self.get_corrected_pose(raw_odom_pose, candidate)
            s = self.score(lidar, cor_pose)

        # 5. Mettre à jour si amélioration
            if s > best_score:
                best_score = s
                best_ref = candidate
                N = 0
            else:
                N = N + 1
        # 6. Mémoriser la meilleure référence trouvée

        self.odom_pose_ref = best_ref
        return best_score


    def update_map(self, lidar, pose):
        """
        Bayesian map update with new observation
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, corrected pose in world coordinates
        """
        ranges  = lidar.get_sensor_values()
        angles  = lidar.get_ray_angles()

        # 1. Filtrer les rayons à la portée maximale (pas d'obstacle réel)
        max_range = 600  # ou 600 par défaut
        valid     = ranges < max_range - 1e-3
        r_used    = ranges[valid]
        a_used    = angles[valid]

        # 2. Coordonnées des impacts dans le repère absolu
        x_obs = pose[0] + r_used * np.cos(a_used + pose[2])
        y_obs = pose[1] + r_used * np.sin(a_used + pose[2])

        # 3. Zone à proba 0.5 : on ne touche pas la cellule juste avant l'impact
        eps        = 5.0  # unités-monde (constante claire, ~2-3 cellules)
        r_free_end = np.maximum(r_used - eps, 0)
        x_free_end = pose[0] + r_free_end * np.cos(a_used + pose[2])
        y_free_end = pose[1] + r_free_end * np.sin(a_used + pose[2])

        # 4. Espace libre : Bresenham de la pose au point « free_end »
        val_libre   = -1.0   # log-odd faible (négatif)
        val_obstacle = +2.0  # log-odd fort (positif)

        for i in range(len(x_free_end)):
            self.grid.add_value_along_line(pose[0], pose[1],
                                            x_free_end[i], y_free_end[i],
                                            val_libre)

        # 5. Obstacle : on tape juste sur les cellules d'impact (pas de décalage diagonal)
        self.grid.add_map_points(x_obs, y_obs, val_obstacle)

        # 6. Seuillage pour éviter les divergences (consigne explicite)
        self.grid.occupancy_map = np.clip(self.grid.occupancy_map, -40, 40)


    def compute(self):
        """ Useless function, just for the exercise on using the profiler """
        # Remove after TP1

        ranges = np.random.rand(3600)
        ray_angles = np.arange(-np.pi, np.pi, np.pi / 1800)

        # Poor implementation of polar to cartesian conversion
        points = []
        for i in range(3600):
            pt_x = ranges[i] * np.cos(ray_angles[i])
            pt_y = ranges[i] * np.sin(ray_angles[i])
            points.append([pt_x, pt_y])
