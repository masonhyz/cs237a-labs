#!/usr/bin/env python3

import rclpy                    # ROS2 client library
from rclpy.node import Node     # ROS2 node baseclass
from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan, StochOccupancyGrid2D, OccupancyGrid
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_lib.tf_utils import quaternion_to_yaw
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
from std_msgs.msg import Bool
import numpy as np
from numpy import linalg
import typing as T
from scipy.signal import convolve2d

class Explorer(Node):
    def __init__(self):
        
        super().__init__("explorer")
        self.occupancy_grid = None
        self.state = None
        self.image_detected = False
        self.nav_success_sub = self.create_subscription(Bool, "/nav_success", self.explore, 10)
        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self.update_occupancy, 10)
        self.state_sub = self.create_subscription(TurtleBotState, "/state", self.state_callback, 10)
        self.cmd_nav_pub = self.create_publisher(TurtleBotState, "/cmd_nav", 10)
        self.detect_image_sub = self.create_subscription(Bool, "/detector_bool", self.detect_callback, 10)

    def detect_callback(self, msg: Bool):
        if msg.data:
            self.image_detected = True

    def update_occupancy(self, msg: OccupancyGrid):
        self.occupancy_grid = StochOccupancyGrid2D(
            resolution=msg.info.resolution,
            size_xy=np.array([msg.info.width, msg.info.height]),
            origin_xy=np.array([msg.info.origin.position.x, msg.info.origin.position.y]),
            window_size=9,
            probs=msg.data,
        )
        self.get_logger().info("Updated Map")
    
    def state_callback(self, msg: TurtleBotState):
        self.state = msg
        if self.image_detected:
            self.cmd_nav_pub.publish(self.state)
        # self.get_logger().info("Updated state")
        
    def explore(self, msg: Bool):
        if msg:
            self.get_logger().info("Exploring frontier")
            next_goal = self.compute_next()
            if next_goal is not None:
                self.get_logger().info("Found frontier!")
                self.cmd_nav_pub.publish(next_goal)
            else:
                self.get_logger().info("Nothing found")
                
            
           
    def compute_next(self):
        if self.state is None or self.occupancy_grid is None:
            self.get_logger().warn("Unable to find current state and occupancy map yet")
            return None

        current_state = np.array([self.state.x, self.state.y])
        window_size = 13 
        twenty_percent=0.2*window_size*window_size
        thirty_percent=0.3*window_size*window_size
        window=np.ones((window_size,window_size))
        count_occupied=convolve2d((self.occupancy_grid.probs>=0.5).astype(int),window,mode='same',boundary='fill',fillvalue=0)
        count_unoccupied=convolve2d(((self.occupancy_grid.probs>=0)&(self.occupancy_grid.probs<0.5)).astype(int),window,mode='same',boundary='fill',fillvalue=0)
        count_unknown=convolve2d((self.occupancy_grid.probs==-1).astype(int),window,mode='same', boundary='fill',fillvalue=0)
        self.get_logger().info("ABHIJNYAAAAAAAAAAA")
        self.get_logger().info(f"Prob matrix size: {self.occupancy_grid.probs.shape}")
        self.get_logger().info(f"count_unknown size: {count_unknown.shape}")
        
        self.get_logger().info(f"size x y : {self.occupancy_grid.size_xy}")
        
        
        
        frontier_states=[]
        for y in range(self.occupancy_grid.size_xy[1]):
            for x in range(self.occupancy_grid.size_xy[0]):
                if count_unknown [y][x]>=twenty_percent and count_occupied[y][x]==0 and count_unoccupied[y][x]>=thirty_percent:
                    frontier_states.append(self.occupancy_grid.grid2state(np.array([x,y])))
        
        if len(frontier_states) == 0:
            self.get_logger().info("No frontier found")
            return None

        frontier_states=np.array(frontier_states)
        distances=np.linalg.norm(frontier_states-current_state,axis=1)
                
        closest_idx = np.argmin(distances)
        
        next_frontier = TurtleBotState()
        next_frontier.x = frontier_states[closest_idx][0]
        next_frontier.y = frontier_states[closest_idx][1]
        dx = frontier_states[closest_idx][0] - self.state.x
        dy = frontier_states[closest_idx][1] - self.state.y
        next_frontier.theta = np.arctan2(dy, dx)
        return next_frontier
        

if __name__ == "__main__":
    rclpy.init()            # initialize ROS client library
    node = Explorer()    # create the node instance
    rclpy.spin(node)        # call ROS2 default scheduler
    rclpy.shutdown()        # clean up after node exit