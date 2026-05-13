"""
Robot controller definition
Complete controller including SLAM, planning, path following
"""
import numpy as np

from place_bot.simulation.robot.robot_abstract import RobotAbstract
from place_bot.simulation.robot.odometer import OdometerParams
from place_bot.simulation.ray_sensors.lidar import LidarParams

from tiny_slam import TinySlam

from control import potential_field_control, reactive_obst_avoid, dwa_control, potential_field_control_2
from occupancy_grid import OccupancyGrid
from planner import Planner


# Definition of our robot controller
class MyRobotSlam(RobotAbstract):
    """A robot controller including SLAM, path planning and path following"""

    def __init__(self,
                 lidar_params: LidarParams = LidarParams(),
                 odometer_params: OdometerParams = OdometerParams()):
        # Passing parameter to parent class
        super().__init__(lidar_params=lidar_params,
                         odometer_params=odometer_params)

        # step counter to deal with init and display
        self.counter = 0

        # Init SLAM object
        # Here we cheat to get an occupancy grid size that's not too large, by using the
        # robot's starting position and the maximum map size that we shouldn't know.
        size_area = (1400, 1000)
        robot_position = (439.0, 195)
        self.occupancy_grid = OccupancyGrid(x_min=-(size_area[0] / 2 + robot_position[0]),
                                            x_max=size_area[0] / 2 - robot_position[0],
                                            y_min=-(size_area[1] / 2 + robot_position[1]),
                                            y_max=size_area[1] / 2 - robot_position[1],
                                            resolution=2)

        self.tiny_slam = TinySlam(self.occupancy_grid)
        self.planner = Planner(self.occupancy_grid)

        # storage for pose after localization
        self.corrected_pose = np.array([0, 0, 0])

        self.goal       = None      # objectif courant (3D)
        self.path       = None    # trajectoire calculée par A* (2×N)
        self.path_index = 0    # index du waypoint courant
        self.waypoints  = self.occupancy_grid.get_predefined_waypoints_v2(M=3)
        self.goal_index  = 0
        self.exploration_done = False  # vrai dès que tous les waypoints sont atteints
        self.score_threshold   = 0    # seuil minimal pour faire confiance à localise
        self.return_goal = np.array([0.0, 0.0, 0.0])  # point de retour (position initiale)
        self.arrived = False     # flag de fin de mission
        self.waypoint_tol = 30.0     # distance pour considérer un waypoint atteint
        self.pf_state = {} # pour la mémoire dans la version 2 du control de potentiel

    def control(self):
        """
        Main control function executed at each time step

        """
        # raw_odom_pose = self.odometer_values()
        # score = self.tiny_slam.localise(self.lidar(), raw_odom_pose=raw_odom_pose)
        # print(score)
        return self.control_tp2()

    def control_tp1(self):
        """
        Control function for TP1
        Control funtion with minimal random motion
        """
        # self.tiny_slam.compute()
        pose = self.odometer_values()

        # Compute new command speed to perform obstacle avoidance
        command = reactive_obst_avoid(self.lidar())
        self.tiny_slam.update_map(self.lidar(), pose)
        self.occupancy_grid.display_cv(pose)
        return command

    def control_tp2(self):
        """
        Control function for TP2
        Main control function with full SLAM, random exploration and path planning
        """

        self.counter = self.counter + 1
        raw_odom = self.odometer_values()

        if self.counter < 40:
            cor_pose = self.tiny_slam.get_corrected_pose(raw_odom)
            self.tiny_slam.update_map(self.lidar(), cor_pose)
            
            if self.counter % 10 == 0:
                self.occupancy_grid.display_cv(cor_pose)
                
            return {'forward': 0.0, 'rotation': 0.0}

        #  PHASE 1 : exploration (cartographie + localisation)
        #  Termine dès que TOUS les waypoints prédéfinis ont été visités
        if not self.exploration_done:

            # 1.1. Localisation après quelques itérations (laisser la carte se former)
            if self.counter > 50:
                score = self.tiny_slam.localise(self.lidar(), raw_odom)
            else:
                score = np.inf   # on accepte toujours pendant le warmup

            # 1.2. Pose corrigée (servira au goal management et à la mise à jour carte)
            cor_pose = self.tiny_slam.get_corrected_pose(raw_odom)

            # 1.3. Mise à jour de la carte
            if score >= self.score_threshold:
                self.tiny_slam.update_map(self.lidar(), cor_pose)

            # 1.4. Gestion des waypoints (parcours unique, pas de cycle infini)
            if self.goal is None:
                self.goal = self.waypoints[self.goal_index]
                print(f'[Phase 1] Cible waypoint {self.goal_index+1}/{len(self.waypoints)} : {self.goal[:2]}')
            elif np.linalg.norm(self.goal[0:2] - cor_pose[0:2]) < self.waypoint_tol:
                # waypoint courant atteint
                self.goal_index += 1
                if self.goal_index >= len(self.waypoints):
                    # tous les waypoints visités → fin de l'exploration
                    self.exploration_done = True
                    print(f'[Phase 1] Tous les waypoints atteints (itération {self.counter}) → planification A*')
                else:
                    self.goal = self.waypoints[self.goal_index]
                    print(f'[Phase 1] Cible waypoint {self.goal_index+1}/{len(self.waypoints)} : {self.goal[:2]}')

            # 1.5. Si la phase 1 vient juste de se terminer, on ne renvoie pas de commande ici :
            #      on laisse couler vers la phase 2 dans la même itération.
            if not self.exploration_done:
                # command = potential_field_control(self.lidar(), cor_pose, self.goal)
                # command, self.pf_state = potential_field_control_2(self.lidar(), cor_pose, self.goal, self.pf_state)
                command = dwa_control(self.lidar(), cor_pose, self.goal)
                # 1.6. Affichage
                if self.counter % 10 == 0:
                    self.occupancy_grid.display_cv(cor_pose, goal=self.goal, traj=self.path)

                return command

        #  PHASE 2 : calcul du chemin retour (une seule fois)
        if self.path is None:
            cor_pose = self.tiny_slam.get_corrected_pose(raw_odom)
            start    = cor_pose
            print(f'[Phase 2] Planification A* de {start[:2]} vers {self.return_goal[:2]}')
            self.path = self.planner.plan(start, self.return_goal)
            self.path_index = 0
            if self.path is None:
                print('Pas de chemin trouvé, on retente à la prochaine itération')
                return {'forward': 0.0, 'rotation': 0.0}
            print(f'[Phase 2] Chemin trouvé : {self.path.shape[1]} waypoints')
            # Affichage immédiat du chemin calculé (avant de commencer à le suivre)
            self.occupancy_grid.display_cv(cor_pose,
                                           goal=self.return_goal,
                                           traj=self.path)

        #  PHASE 3 : suivi du chemin
        cor_pose = self.tiny_slam.get_corrected_pose(raw_odom)
        # On continue à mettre la carte à jour pendant le retour
        self.tiny_slam.update_map(self.lidar(), cor_pose)

        # 3.1. Critère d'arrêt global : robot revenu près du point initial
        dist_to_home = np.linalg.norm(cor_pose[0:2] - self.return_goal[0:2])
        if dist_to_home < 20.0:
            if not self.arrived:
                print(f'[Phase 3] Robot revenu au point initial (d={dist_to_home:.1f})')
                self.arrived = True
            self.occupancy_grid.display_cv(cor_pose,
                                           goal=self.return_goal,
                                           traj=self.path)
            return {'forward': 0.0, 'rotation': 0.0}

        # 3.2. Avancer le waypoint si on est suffisamment proche
        while self.path_index < self.path.shape[1]:
            wp = self.path[:, self.path_index]
            if np.linalg.norm(wp - cor_pose[0:2]) < 20:
                self.path_index = self.path_index + 1
            else:
                break

        # 3.3. Si tous les waypoints traités mais robot pas encore arrivé : cibler l'origine
        if self.path_index >= self.path.shape[1]:
            wp_world = self.return_goal
        else:
            # 3.4. Champ de potentiel vers le waypoint courant
            wp_world = np.array([self.path[0, self.path_index],
                                 self.path[1, self.path_index],
                                 0.0])

        # command = potential_field_control(self.lidar(), cor_pose, wp_world)
        # command, self.pf_state = potential_field_control_2(self.lidar(), cor_pose, wp_world, self.pf_state)
        command = dwa_control(self.lidar(), cor_pose, wp_world)
        
        # 3.5. Affichage
        if self.counter % 10 == 0:
            self.occupancy_grid.display_cv(cor_pose, goal=wp_world, traj=self.path)
        return command


