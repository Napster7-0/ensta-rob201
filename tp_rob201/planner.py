"""
Planner class
Implementation of A*
"""

import copy
import heapq
import math
from collections import defaultdict
from typing import Tuple

import cv2
import numpy as np
from occupancy_grid import OccupancyGrid


class Planner:
    """Simple occupancy grid Planner"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid
        self.map_walls = None

    def get_neighbors(self, current_cell):
        """ Return list of free (i.e. not obstacle) neighbour cells
            with the format of current_cell: (i, j) in the map frame
        """
        neighbor_list = []
        i, j = current_cell

        # Une cellule est considérée comme obstacle dès que le log-odd est > seuil.
        # On s'appuie sur self.map_walls (carte avec murs dilatés) si elle existe.
        OBSTACLE_THRESHOLD = 0.0
        wall_map = self.map_walls if self.map_walls is not None else self.grid.occupancy_map

        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                # bornes de la carte
                if ni < 0 or ni >= self.grid.x_max_map:
                    continue
                if nj < 0 or nj >= self.grid.y_max_map:
                    continue
                # cellule libre uniquement
                if wall_map[ni, nj] > OBSTACLE_THRESHOLD:
                    continue
                neighbor_list.append((ni, nj))
        return neighbor_list

    def heuristic(self, cell_1: Tuple[int, int], cell_2: Tuple[int, int]):
        """ Return heuristic goal distance (euclidean distance in cells) """
        dx = cell_2[0] - cell_1[0]
        dy = cell_2[1] - cell_1[1]
        return math.sqrt(dx * dx + dy * dy)

    def reconstruct_path(self, came_from, goal):
        """ Extract path after cost computation """
        total_path = [goal]
        cell = goal
        while cell in came_from.keys():
            cell = came_from[cell]
            total_path.insert(0, cell)

        total_path = np.array(total_path)
        traj_world_x, traj_world_y = self.grid.conv_map_to_world(total_path[:, 0], total_path[:, 1])
        return np.vstack((traj_world_x, traj_world_y))

    def plan(self, start, goal):
        """
        Compute a path using A*, recompute plan if start or goal change
        start : [x, y, theta] nparray, start pose in world coordinates (theta unused)
        goal : [x, y, theta] nparray, goal pose in world coordinates (theta unused)
        """

        # Conversion monde -> indices grille (forcer en int Python pour les clés de dict)
        sx, sy = self.grid.conv_world_to_map(float(start[0]), float(start[1]))
        gx, gy = self.grid.conv_world_to_map(float(goal[0]), float(goal[1]))
        start: Tuple[int, int] = (int(sx), int(sy))
        goal: Tuple[int, int] = (int(gx), int(gy))

        # Vérification que start et goal sont dans les bornes de la carte
        if not (0 <= start[0] < self.grid.x_max_map and 0 <= start[1] < self.grid.y_max_map):
            print(f'Start {start} hors carte')
            return None
        if not (0 <= goal[0] < self.grid.x_max_map and 0 <= goal[1] < self.grid.y_max_map):
            print(f'Goal {goal} hors carte')
            return None

        # Carte des murs avec marge de sécurité : on dilate les obstacles
        OBSTACLE_THRESHOLD = 0.0
        DILATION_KERNEL = 5  # ~5 cellules = ~10 unités-monde (résolution 2)

        self.map_walls = copy.deepcopy(self.grid.occupancy_map)
        obstacles_mask = (self.grid.occupancy_map > OBSTACLE_THRESHOLD).astype(np.uint8)
        kernel = np.ones((DILATION_KERNEL, DILATION_KERNEL), np.uint8)
        obstacles_dilated = cv2.dilate(obstacles_mask, kernel)
        self.map_walls[obstacles_dilated > 0] = 100.0  # marqueur d'obstacle

        # On force un petit voisinage 3x3 autour de start et goal à être libre,
        # pour que A* puisse démarrer/terminer même si la dilatation a englouti ces cellules
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                si, sj = start[0] + di, start[1] + dj
                gi, gj = goal[0] + di, goal[1] + dj
                if 0 <= si < self.grid.x_max_map and 0 <= sj < self.grid.y_max_map:
                    self.map_walls[si, sj] = -10.0
                if 0 <= gi < self.grid.x_max_map and 0 <= gj < self.grid.y_max_map:
                    self.map_walls[gi, gj] = -10.0

        # cv2.imshow("map_walls", self.map_walls)

        # min heap to contain values to explore next
        open_set = [(0.0, start)]
        heapq.heapify(open_set)

        # dictionary to trace back route
        came_from = {}

        # cost to get to each cell
        g_score = defaultdict(lambda: math.inf)
        g_score[start] = 0.0

        # best guess of cost for each cell (cost + heuristic)
        f_score = defaultdict(lambda: math.inf)
        f_score[start] = self.heuristic(start, goal)

        while len(open_set) > 0:
            current = heapq.heappop(open_set)
            current_f, current_cell = current
            # lazy deletion: skip stale entries
            if current_f > f_score[current_cell]:
                continue
            if current_cell == goal:
                return self.reconstruct_path(came_from, goal)

            neighbours = self.get_neighbors(current_cell)
            for cell in neighbours:
                tentative_g_score = g_score[current_cell] + self.heuristic(current_cell, cell)
                if tentative_g_score < g_score[cell]:
                    # better path, recording it
                    came_from[cell] = current_cell
                    g_score[cell] = tentative_g_score
                    f_score[cell] = tentative_g_score + self.heuristic(cell, goal)
                    heapq.heappush(open_set, (f_score[cell], cell))

        # goal was never reached
        print('failed getting to objective')
        return None

    def explore_frontiers(self):
        """ Frontier based exploration """
        goal = np.array([0, 0, 0])  # frontier to reach for exploration
        return goal
