#!/usr/bin/env python3

import numpy, rclpy

from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState

from std_msgs.msg import Int64, Bool

class PerceptionController(BaseHeadingController):
    """ Student can inherit from this class to build a heading controller node

    This node takes target pose from /cmd_pose, and control the robot's orientation
    towards the target pose orientation using a heading controller
    """

    def __init__(self, node_name: str = "heading_control") -> None:
        super().__init__(node_name)
        self.image_detected = False
        self.detect_image_sub = self.create_subscription(Bool, "/detector_bool", self.detect_callback, 10)

    def detect_callback(self, msg: Bool):
        self.image_detected = msg.data

    def compute_control_with_goal(self,
        state: TurtleBotState,
        goal: TurtleBotState
    ) -> TurtleBotControl:
        """ Compute control given current robot state and goal state

        Args:
            state (TurtleBotState): current robot state
            goal (TurtleBotState): current goal state

        Returns:
            TurtleBotControl: control command
        """
        command = TurtleBotControl()
        omega = 0.2 if self.image_detected else 0
        command.v = 0.0
        command.omega = float(omega)
        return command

if __name__ == "__main__":
    rclpy.init()        # initialize ROS2 context (must run before any other rclpy call)
    node = PerceptionController()  # instantiate the heartbeat node
    rclpy.spin(node)    # Use ROS2 built-in schedular for executing the node
    rclpy.shutdown()    # cleanly shutdown ROS2 context
    