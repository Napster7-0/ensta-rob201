"""
Occupancy grid class
Includes initialisation, raytracing, display...
"""

import pickle

import cv2
import numpy as np
from matplotlib import pyplot as plt

VIDEO_OUT = False


class OccupancyGrid:
    """Simple occupancy grid"""

    def __init__(self, x_min, x_max, y_min, y_max, resolution):
        # Given : constructor
        self.x_min_world = x_min
        self.x_max_world = x_max
        self.y_min_world = y_min
        self.y_max_world = y_max
        self.resolution = resolution

        self.x_max_map, self.y_max_map = self.conv_world_to_map(
            self.x_max_world, self.y_max_world)

        self.occupancy_map = np.zeros(
            (int(self.x_max_map), int(self.y_max_map)))

        if VIDEO_OUT:
            self.cv_out = cv2.VideoWriter('rob201.avi',
                                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                          7,
                                          (self.x_max_map, self.y_max_map))

    def conv_world_to_map(self, x_world, y_world):
        """
        Convert from world coordinates to map coordinates (i.e. cell index in the grid map)
        x_world, y_world : list of x and y coordinates in m
        """

        x_map = (x_world - self.x_min_world) / self.resolution
        y_map = (y_world - self.y_min_world) / self.resolution

        if isinstance(x_map, float):
            x_map = int(x_map)
            y_map = int(y_map)
        elif isinstance(x_map, np.ndarray):
            x_map = x_map.astype(int)
            y_map = y_map.astype(int)

        return x_map, y_map

    def conv_map_to_world(self, x_map, y_map):
        """
        Convert from map coordinates to world coordinates
        x_map, y_map : list of x and y coordinates in cell numbers (~pixels)
        """

        x_world = self.x_min_world + x_map * self.resolution
        y_world = self.y_min_world + y_map * self.resolution

        if isinstance(x_world, np.ndarray):
            x_world = x_world.astype(float)
            y_world = y_world.astype(float)

        return x_world, y_world

    def add_value_along_line(self, x_0: float, y_0: float, x_1: float, y_1: float, val):
        """
        Add a value to a line of points using Bresenham algorithm, input in world coordinates
        x_0, y_0 : starting point coordinates in m
        x_1, y_1 : end point coordinates in m
        val : value to add to each cell of the line
        """

        # convert to pixels
        x_start, y_start = self.conv_world_to_map(x_0, y_0)
        x_end, y_end = self.conv_world_to_map(x_1, y_1)

        if x_start < 0 or x_start >= self.x_max_map or y_start < 0 or y_start >= self.y_max_map:
            return

        if x_end < 0 or x_end >= self.x_max_map or y_end < 0 or y_end >= self.y_max_map:
            return

        # Bresenham line drawing
        d_x = x_end - x_start
        d_y = y_end - y_start
        is_steep = abs(d_y) > abs(d_x)  # determine how steep the line is
        if is_steep:  # rotate line
            x_start, y_start = y_start, x_start
            x_end, y_end = y_end, x_end
        # swap start and end points if necessary and store swap state
        if x_start > x_end:
            x_start, x_end = x_end, x_start
            y_start, y_end = y_end, y_start
        d_x = x_end - x_start  # recalculate differentials
        d_y = y_end - y_start  # recalculate differentials
        error = int(d_x / 2.0)  # calculate error
        y_step = 1 if y_start < y_end else -1
        # iterate over bounding box generating points between start and end
        y = y_start
        points = []
        for x in range(x_start, x_end + 1):
            coord = [y, x] if is_steep else [x, y]
            points.append(coord)
            error -= abs(d_y)
            if error < 0:
                y += y_step
                error += d_x
        points = np.array(points).T

        # add value to the points
        self.occupancy_map[points[0], points[1]] += val

    def add_map_points(self, points_x, points_y, val):
        """
        Add a value to an array of points, input coordinates in meters
        points_x, points_y :  list of x and y coordinates in m
        val :  value to add to the cells of the points
        """

        x_px, y_px = self.conv_world_to_map(points_x, points_y)

        select = np.logical_and(np.logical_and(x_px >= 0, x_px < self.x_max_map),
                                np.logical_and(y_px >= 0, y_px < self.y_max_map))
        x_px = x_px[select]
        y_px = y_px[select]

        self.occupancy_map[x_px, y_px] += val

    def get_predefined_waypoints(self, K=2, M=4, step=500, seed=1):
        """
        Génère M waypoints situés sur un même polynôme d'interpolation.
        Les K ancres du polynôme sont tirées par marche aléatoire bornée
        (chaque ancre est définie relativement à la précédente).
        K : nombre d'ancres
        M : nombre de waypoints échantillonnés sur le polynôme
        step : amplitude max de l'incrément entre deux ancres (unités-monde)
        seed : pour reproductibilité (None = aléatoire)
        """
        rng = np.random.default_rng(seed)

        margin_x = 0.15 * (self.x_max_world - self.x_min_world)
        margin_y = 0.15 * (self.y_max_world - self.y_min_world)
        x_lo, x_hi = self.x_min_world + margin_x, self.x_max_world - margin_x
        y_lo, y_hi = self.y_min_world + margin_y, self.y_max_world - margin_y

        # 1. Tirage relatif des ancres (random walk borné)
        ax = [0.0]
        ay = [0.0]
        for _ in range(1, K):
            nx = float(np.clip(ax[-1] + rng.uniform(-step, step), x_lo, x_hi))
            ny = float(np.clip(ay[-1] + rng.uniform(-step, step), y_lo, y_hi))
            ax.append(nx)
            ay.append(ny)

        # 2. Un seul polynôme d'interpolation (Lagrange via polyfit, degré K-1)
        t_anch = np.linspace(0.0, 1.0, K)
        cx = np.polyfit(t_anch, ax, deg=K - 1)
        cy = np.polyfit(t_anch, ay, deg=K - 1)

        # 3. Échantillonnage successif des M waypoints sur ce polynôme
        t_samples = np.linspace(0.0, 1.0, M)
        xs = np.polyval(cx, t_samples)
        ys = np.polyval(cy, t_samples)

        return [np.array([float(xs[i]), float(ys[i]), 0.0]) for i in range(M)]
    

    def display_plt(self, robot_pose, goal=None, traj=None):
        """
        Screen display of map and robot pose,
        using matplotlib (slower than the opencv version)
        robot_pose : [x, y, theta] nparray, corrected robot pose
        """

        plt.cla()
        plt.imshow(self.occupancy_map.T, origin='lower',
                   extent=[self.x_min_world, self.x_max_world, self.y_min_world, self.y_max_world])
        plt.clim(-4, 4)
        plt.axis("equal")

        if traj is not None:
            plt.plot(traj[0, :], traj[1, :], 2, 'w')

        if goal is not None:
            plt.scatter(goal[0], goal[1], 4, 'white')

        delta_x = np.cos(robot_pose[2]) * 10
        delta_y = np.sin(robot_pose[2]) * 10
        plt.arrow(robot_pose[0], robot_pose[1], delta_x, delta_y,
                  color='red', head_width=5, head_length=10, )

        # plt.show()
        plt.pause(0.001)

    def display_cv(self, robot_pose, goal=None, traj=None):
        """
        Screen display of map and robot pose,
        using opencv (faster than the matplotlib version)
        robot_pose : [x, y, theta] nparray, corrected robot pose
        """

        img = cv2.flip(self.occupancy_map.T, 0)
        img = img - img.min()
        img = img / img.max() * 255
        img = np.uint8(img)
        img_color = cv2.applyColorMap(src=img, colormap=cv2.COLORMAP_JET)

        if traj is not None:
            traj_map_x, traj_map_y = self.conv_world_to_map(traj[0, :], traj[1, :])
            traj_map = np.vstack((traj_map_x, self.y_max_map - traj_map_y))
            for i in range(len(traj_map_x) - 1):
                cv2.line(img_color, traj_map[:, i], traj_map[:, i + 1], (180, 180, 180), 2)

        if goal is not None:
            pt_x, pt_y = self.conv_world_to_map(goal[0], goal[1])
            point = (int(pt_x), self.y_max_map - int(pt_y))
            color = (255, 255, 255)
            cv2.circle(img_color, point, 3, color, -1)

        pt2_x = robot_pose[0] + np.cos(robot_pose[2]) * 20
        pt2_y = robot_pose[1] + np.sin(robot_pose[2]) * 20
        pt2_x, pt2_y = self.conv_world_to_map(pt2_x, pt2_y)

        pt1_x, pt1_y = self.conv_world_to_map(robot_pose[0], robot_pose[1])

        # print("robot_pose", robot_pose)
        pt1 = (int(pt1_x), self.y_max_map - int(pt1_y))
        pt2 = (int(pt2_x), self.y_max_map - int(pt2_y))
        cv2.arrowedLine(img=img_color, pt1=pt1, pt2=pt2,
                        color=(0, 0, 255), thickness=2)
        cv2.imshow("map slam", img_color)
        if VIDEO_OUT:
            self.cv_out.write(img_color)

        cv2.waitKey(1)

    def save(self, filename):
        """
        Save map as image and pickle object
        filename : base name (without extension) of file on disk
        """

        plt.imshow(self.occupancy_map.T, origin='lower',
                   extent=[self.x_min_world, self.x_max_world,
                           self.y_min_world, self.y_max_world])
        plt.clim(-4, 4)
        plt.axis("equal")
        plt.savefig(filename + '.png')

        with open(filename + ".p", "wb") as fid:
            pickle.dump({'occupancy_map': self.occupancy_map,
                         'resolution': self.resolution,
                         'x_min_world': self.x_min_world,
                         'x_max_world': self.x_max_world,
                         'y_min_world': self.y_min_world,
                         'y_max_world': self.y_max_world}, fid)

        if VIDEO_OUT:
            self.cv_out.release()

    def load(self, filename):
        """
        Load map from pickle object
        filename : base name (without extension) of file on disk
        """
        # TODO
