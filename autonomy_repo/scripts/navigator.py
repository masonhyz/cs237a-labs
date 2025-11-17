#!/usr/bin/env python3
from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState

import rclpy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import splrep, splev
from numpy import linalg


class AStar(object):
    """Represents a motion planning problem to be solved using A*"""

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution=1):
        self.statespace_lo = statespace_lo         # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = statespace_hi         # state space upper bound (e.g., [5, 5])
        self.occupancy = occupancy                 # occupancy grid (a DetOccupancyGrid2D object)
        self.resolution = resolution               # resolution of the discretization of state space (cell/m)
        self.x_offset = x_init                     
        self.x_init = self.snap_to_grid(x_init)    # initial state
        self.x_goal = self.snap_to_grid(x_goal)    # goal state

        self.closed_set = set()    # the set containing the states that have been visited
        self.open_set = set()      # the set containing the states that are condidate for future expension

        self.est_cost_through = {}  # dictionary of the estimated cost from start to goal passing through state (often called f score)
        self.cost_to_arrive = {}    # dictionary of the cost-to-arrive at state from start (often called g score)
        self.came_from = {}         # dictionary keeping track of each state's parent to reconstruct the path

        self.open_set.add(self.x_init)
        self.cost_to_arrive[self.x_init] = 0
        self.est_cost_through[self.x_init] = self.distance(self.x_init,self.x_goal)

        self.path = None        # the final path as a list of states

    def is_free(self, x):
        """
        Checks if a give state x is free, meaning it is inside the bounds of the map and
        is not inside any obstacle.
        Inputs:
            x: state tuple
        Output:
            Boolean True/False
        Hint: self.occupancy is a DetOccupancyGrid2D object, take a look at its methods for what might be
              useful here
        """
        ########## Code starts here ##########
        x = np.array(x, dtype=float)
        return self.occupancy.is_free(x)
        ########## Code ends here ##########

    def distance(self, x1, x2):
        """
        Computes the Euclidean distance between two states.
        Inputs:
            x1: First state tuple
            x2: Second state tuple
        Output:
            Float Euclidean distance

        HINT: This should take one line. Tuples can be converted to numpy arrays using np.array().
        """
        ########## Code starts here ##########
        return np.linalg.norm(np.array(x1) - np.array(x2))
        ########## Code ends here ##########

    def snap_to_grid(self, x):
        """ Returns the closest point on a discrete state grid
        Input:
            x: tuple state
        Output:
            A tuple that represents the closest point to x on the discrete state grid
        """
        return (
            self.resolution * round((x[0] - self.x_offset[0]) / self.resolution) + self.x_offset[0],
            self.resolution * round((x[1] - self.x_offset[1]) / self.resolution) + self.x_offset[1],
        )

    def get_neighbors(self, x):
        """
        Gets the FREE neighbor states of a given state x. Assumes a motion model
        where we can move up, down, left, right, or along the diagonals by an
        amount equal to self.resolution.
        Input:
            x: tuple state
        Ouput:
            List of neighbors that are free, as a list of TUPLES

        HINTS: Use self.is_free to check whether a given state is indeed free.
               Use self.snap_to_grid (see above) to ensure that the neighbors
               you compute are actually on the discrete grid, i.e., if you were
               to compute neighbors by adding/subtracting self.resolution from x,
               numerical errors could creep in over the course of many additions
               and cause grid point equality checks to fail. To remedy this, you
               should make sure that every neighbor is snapped to the grid as it
               is computed.
        """
        neighbors = []
        ########## Code starts here ##########
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    continue
                nb = (x[0] + i * self.resolution, x[1] + j * self.resolution)
                nb = self.snap_to_grid(nb)
                if self.is_free(nb):
                    neighbors.append(nb)
        ########## Code ends here ##########
        return neighbors

    def find_best_est_cost_through(self):
        """
        Gets the state in open_set that has the lowest est_cost_through
        Output: A tuple, the state found in open_set that has the lowest est_cost_through
        """
        return min(self.open_set, key=lambda x: self.est_cost_through[x])

    def reconstruct_path(self):
        """
        Use the came_from map to reconstruct a path from the initial location to
        the goal location
        Output:
            A list of tuples, which is a list of the states that go from start to goal
        """
        path = [self.x_goal]
        current = path[-1]
        while current != self.x_init:
            path.append(self.came_from[current])
            current = path[-1]
        return list(reversed(path))

    def plot_path(self, fig_num=0, show_init_label=True):
        """Plots the path found in self.path and the obstacles"""
        if not self.path:
            return

        self.occupancy.plot(fig_num)

        solution_path = np.asarray(self.path)
        plt.plot(solution_path[:,0],solution_path[:,1], color="green", linewidth=2, label="A* solution path", zorder=10)
        plt.scatter([self.x_init[0], self.x_goal[0]], [self.x_init[1], self.x_goal[1]], color="green", s=30, zorder=10)
        if show_init_label:
            plt.annotate(r"$x_{init}$", np.array(self.x_init) + np.array([.2, .2]), fontsize=16)
        plt.annotate(r"$x_{goal}$", np.array(self.x_goal) + np.array([.2, .2]), fontsize=16)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)

        plt.axis([0, self.occupancy.width, 0, self.occupancy.height])

    def plot_tree(self, point_size=15):
        plot_line_segments([(x, self.came_from[x]) for x in self.open_set if x != self.x_init], linewidth=1, color="blue", alpha=0.2)
        plot_line_segments([(x, self.came_from[x]) for x in self.closed_set if x != self.x_init], linewidth=1, color="blue", alpha=0.2)
        px = [x[0] for x in self.open_set | self.closed_set if x != self.x_init and x != self.x_goal]
        py = [x[1] for x in self.open_set | self.closed_set if x != self.x_init and x != self.x_goal]
        plt.scatter(px, py, color="blue", s=point_size, zorder=10, alpha=0.2)

    def solve(self):
        """
        Solves the planning problem using the A* search algorithm. It places
        the solution as a list of tuples (each representing a state) that go
        from self.x_init to self.x_goal inside the variable self.path
        Input:
            None
        Output:
            Boolean, True if a solution from x_init to x_goal was found

        HINTS:  We're representing the open and closed sets using python's built-in
                set() class. This allows easily adding and removing items using
                .add(item) and .remove(item) respectively, as well as checking for
                set membership efficiently using the syntax "if item in set".
        """
        ########## Code starts here ##########
        while self.open_set:
            x = self.find_best_est_cost_through()
            if x == self.x_goal:
                self.path = self.reconstruct_path()
                return True
            self.open_set.remove(x)
            self.closed_set.add(x)

            for nb in self.get_neighbors(x):
                if nb in self.closed_set:
                    continue
                tentative_cost = self.cost_to_arrive[x] + self.distance(x, nb)
                if nb not in self.open_set:
                    self.open_set.add(nb)
                elif tentative_cost > self.cost_to_arrive[nb]:
                    continue
                self.came_from[nb] = x
                self.cost_to_arrive[nb] = tentative_cost
                self.est_cost_through[nb] = tentative_cost + self.distance(nb, self.x_goal)
            
        return False
        ########## Code ends here ##########

class DetOccupancyGrid2D(object):
    """
    A 2D state space grid with a set of rectangular obstacles. The grid is
    fully deterministic
    """
    def __init__(self, width, height, obstacles):
        self.width = width
        self.height = height
        self.obstacles = obstacles

    def is_free(self, x):
        """Verifies that point is not inside any obstacles by some margin"""
        for obs in self.obstacles:
            if x[0] >= obs[0][0] - self.width * .01 and \
               x[0] <= obs[1][0] + self.width * .01 and \
               x[1] >= obs[0][1] - self.height * .01 and \
               x[1] <= obs[1][1] + self.height * .01:
                return False
        return True

    def plot(self, fig_num=0):
        """Plots the space and its obstacles"""
        fig = plt.figure(fig_num)
        ax = fig.add_subplot(111, aspect='equal')
        for obs in self.obstacles:
            ax.add_patch(
            patches.Rectangle(
            obs[0],
            obs[1][0]-obs[0][0],
            obs[1][1]-obs[0][1],))
        ax.set(xlim=(0,self.width), ylim=(0,self.height))






class NavigatorNode(BaseNavigator):
    def __init__(self, kpx: float, kpy: float, kdx: float, kdy: float, V_max: float = 0.5, om_max: float = 1) -> None:
        super().__init__()
        self.kp = 2
        
        self.V_PREV_THRES = 0.0001

        self.kpx = kpx
        self.kpy = kpy
        self.kdx = kdx
        self.kdy = kdy

        self.coeffs = np.zeros(8) # Polynomial coefficients for x(t) and y(t) as
                                 # returned by the differential flatness code

    def reset(self) -> None:
        self.V_prev = 0.
        self.om_prev = 0.
        self.t_prev = 0.
        
    def compute_heading_control(self, state_cur: TurtleBotState, state_des: TurtleBotState) -> TurtleBotControl:
        command = TurtleBotControl()
        error_angle = wrap_angle(state_des.theta - state_cur.theta)
        omega = self.kp*error_angle
        command.v = 0.0
        command.omega = float(omega)
        return command

    def compute_trajectory_tracking_control(self, state: TurtleBotState, plan: TrajectoryPlan, t: float) -> TurtleBotControl:
        tck_x = plan.path_x_spline
        tck_y = plan.path_y_spline
        x_d, xd_d, xdd_d = [splev(t, tck_x, der=i) for i in range(3)]
        y_d, yd_d, ydd_d = [splev(t, tck_y, der=i) for i in range(3)]

        goal_dist = np.hypot(x_d - state.x, y_d - state.y)
        if goal_dist < 0.05:
            control = TurtleBotControl()
            control.v = 0.0
            control.omega = 0.0
            return control

        dt = t - self.t_prev

        if abs(self.V_prev) < self.V_PREV_THRES:
            self.V_prev = self.V_PREV_THRES

        xd = self.V_prev*np.cos(state.theta)
        yd = self.V_prev*np.sin(state.theta)

        # compute virtual controls
        u = np.array([xdd_d + self.kpx*(x_d-state.x) + self.kdx*(xd_d-xd),
                      ydd_d + self.kpy*(y_d-state.y) + self.kdy*(yd_d-yd)])

        # compute real controls
        J = np.array([[np.cos(state.theta), -self.V_prev*np.sin(state.theta)],
                          [np.sin(state.theta), self.V_prev*np.cos(state.theta)]])
        a, om = linalg.solve(J, u)
        V = self.V_prev + a*dt
        
        self.t_prev = t
        self.V_prev = V
        self.om_prev = om

        control = TurtleBotControl()
        control.v = V
        control.omega = om
        # wrap into TurtleBotControl
        return control

    def compute_trajectory_plan(self, state, goal, occupancy, resolution, horizon):
        v_desired = 0.15
        spline_alpha = 0.05

        # --- Extract start and goal as numpy arrays ---
        state_xy = np.array([state.x, state.y])
        goal_xy = np.array([goal.x, goal.y])

        # --- Get map bounds and resolution ---
        lo = occupancy.origin_xy
        hi = occupancy.origin_xy + occupancy.size_xy
        res = getattr(occupancy, "resolution", 0.05)

        print(f"[DEBUG] state_xy={state_xy}, goal_xy={goal_xy}")
        print(f"[DEBUG] map bounds={lo}->{hi}, res={res}")
        print(f"[DEBUG] start free={occupancy.is_free(state_xy)}, goal free={occupancy.is_free(goal_xy)}")

        # --- Run A* ---
        astar = AStar(lo, hi, state_xy, goal_xy, occupancy, resolution=res)
        if not astar.solve() or len(astar.path) < 4:
            print("No path found or path too short")
            return None

        # --- Generate smooth trajectory ---
        self.reset()
        path = np.array(astar.path)

        distances = np.linalg.norm(np.diff(path, axis=0), axis=1)
        times = distances / v_desired
        ts = np.insert(np.cumsum(times), 0, 0.0)
        path_x_spline = splrep(ts, path[:, 0], s=spline_alpha)
        path_y_spline = splrep(ts, path[:, 1], s=spline_alpha)

        return TrajectoryPlan(
            path=path,
            path_x_spline=path_x_spline,
            path_y_spline=path_y_spline,
            duration=ts[-1],
        )


        







if __name__ == "__main__":
    rclpy.init()            # initialize ROS client library
    node = NavigatorNode(
        kpx=2.0,
        kpy=2.0,
        kdx=0.5,
        kdy=0.5,
        V_max=0.5,
        om_max=1.0
    )    # create the node instance
    rclpy.spin(node)        # call ROS2 default scheduler
    rclpy.shutdown()        # clean up after node exits
    