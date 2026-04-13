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


    def _score(self, lidar, pose):
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
        corrected_pose = np.zeros(3)
        odom_pose_ref = self.odom_pose_ref
        d = np.linalg.norm(odom_pose[0:2])
        alpha = np.atan2(odom_pose[1], odom_pose[0])
        corrected_pose[0] = odom_pose_ref[0] + d*np.cos(odom_pose_ref[2] + alpha)
        corrected_pose[1] = odom_pose_ref[1] + d*np.sin(odom_pose_ref[2] + alpha)
        corrected_pose[2] = odom_pose_ref[2] + odom_pose[2]

        return corrected_pose

    def localise(self, lidar, raw_odom_pose):
        """
        Compute the robot position wrt the map, and updates the odometry reference
        lidar : placebot object with lidar data
        odom : [x, y, theta] nparray, raw odometry position
        """
        # TODO for TP4
        N = 0
        sigma = 0.5
        best_score = self._score(lidar,self.get_corrected_pose(raw_odom_pose))
        while N < 10 :
            offset = np.random.normal(loc=0.0, scale=np.sqrt(sigma), size=2)  # bruit sur x,y
            ref = self.odom_pose_ref[:2] + offset
            score = self._score(lidar, self.get_corrected_pose(raw_odom_pose, ref))
            if score > best_score:
                N = 0
                best_score = score
                self.odom_pose_ref=ref
                continue
            N+=1

        return best_score

    def update_map(self, lidar, pose):
        """
        Bayesian map update with new observation
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, corrected pose in world coordinates
        """
        lidar_values = lidar.get_sensor_values()
        lidar_angles = lidar.get_ray_angles()

        # Coordonnées des obstacles dans le repère absolu
        x_obs = pose[0] + lidar_values * np.cos(lidar_angles + pose[2])
        y_obs = pose[1] + lidar_values * np.sin(lidar_angles + pose[2])

        # max_range = lidar_values.max()
        # mask = lidar_values < max_range  # garde seulement les vrais obstacles

        # x_obs = x_obs[mask]
        # y_obs = y_obs[mask]

        eps = 0.01*x_obs.max()
        val_faible = -0.5
        val_forte = 2
        for i in range(x_obs.shape[0]):
            self.grid.add_value_along_line(pose[0], pose[1], x_obs[i], y_obs[i], val_faible)
        self.grid.add_map_points(x_obs, y_obs, val_forte)
        self.grid.add_map_points(x_obs-eps, y_obs-eps, val_forte)
        self.grid.add_map_points(x_obs+eps, y_obs+eps, val_forte)

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
