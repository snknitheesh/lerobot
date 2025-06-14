# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from typing import Any

import numpy as np

from lerobot.common.cameras import make_cameras_from_configs
from lerobot.common.errors import DeviceNotConnectedError
from lerobot.common.model.kinematics import RobotKinematics
from lerobot.common.motors import Motor, MotorNormMode
from lerobot.common.motors.feetech import FeetechMotorsBus

from . import SO101Follower
from .config_so101_follower import SO101FollowerEndEffectorConfig

logger = logging.getLogger(__name__)
EE_FRAME = "gripper_tip"


class SimplePID:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.prev_error = 0
        self.integral = 0

    def step(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

class SO101FollowerEndEffector(SO101Follower):
    """
    SO101Follower robot with end-effector space control.

    This robot inherits from SO101Follower but transforms actions from
    end-effector space to joint space before sending them to the motors.
    """

    config_class = SO101FollowerEndEffectorConfig
    name = "so101_follower_end_effector"

    def __init__(self, config: SO101FollowerEndEffectorConfig):
        super().__init__(config)
        self.pid_x = SimplePID(0.5, 0.01, 0.05)
        self.pid_y = SimplePID(0.5, 0.01, 0.05)
        self.pid_z = SimplePID(0.5, 0.01, 0.05)
        self.last_time = time.time()
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
                "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
                "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
                "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
                "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )

        self.cameras = make_cameras_from_configs(config.cameras)

        self.config = config

        # Initialize the kinematics module for the so101 robot
        self.kinematics = RobotKinematics(robot_type="so_new_calibration")

        # Store the bounds for end-effector position
        self.end_effector_bounds = self.config.end_effector_bounds

        self.current_ee_pos = None
        self.current_joint_pos = None

    @property
    def action_features(self) -> dict[str, Any]:
        """
        Define action features for end-effector control.
        Returns dictionary with dtype, shape, and names.
        """
        return {
            "dtype": "float32",
            "shape": (4,),
            "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
        }

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Transform action from end-effector space to joint space and send to motors.

        Args:
            action: Dictionary with keys 'delta_x', 'delta_y', 'delta_z' for end-effector control
                   or a numpy array with [delta_x, delta_y, delta_z]

        Returns:
            The joint-space action that was sent to the motors
        """
        print("ACTION RECEIVED:", action)
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        joint_keys = [f"{k}.pos" for k in self.bus.motors.keys()]
        if all(k in action for k in joint_keys):
            # Optionally: update current_joint_pos and current_ee_pos
            self.current_joint_pos = np.array([action[k] for k in joint_keys])
            self.current_ee_pos = self.kinematics.forward_kinematics(self.current_joint_pos, frame=EE_FRAME)
            return super().send_action(action)
        pid_flag = True
        # PID smoothing
        if pid_flag:
            now = time.time()
            dt = now - self.last_time if hasattr(self, "last_time") else 0.05
            self.last_time = now

            if isinstance(action, dict) and all(k in action for k in ["delta_x", "delta_y", "delta_z"]):
                # Apply PID to each axis
                action["delta_x"] = self.pid_x.step(action["delta_x"], dt)
                action["delta_y"] = self.pid_y.step(action["delta_y"], dt)
                action["delta_z"] = self.pid_z.step(action["delta_z"], dt)
        
        # Convert action to numpy array if not already
        if isinstance(action, dict):
            if all(k in action for k in ["delta_x", "delta_y", "delta_z"]):
                delta_ee = np.array(
                    [
                        action["delta_x"] * self.config.end_effector_step_sizes["x"],
                        action["delta_y"] * self.config.end_effector_step_sizes["y"],
                        action["delta_z"] * self.config.end_effector_step_sizes["z"],
                    ],
                    dtype=np.float32,
                )
                if "gripper" not in action:
                    action["gripper"] = [1.0]
                action = np.append(delta_ee, action["gripper"])
            else:
                logger.warning(
                    f"Expected action keys 'delta_x', 'delta_y', 'delta_z', got {list(action.keys())}"
                )
                action = np.zeros(4, dtype=np.float32)

        if self.current_joint_pos is None:
            # Read current joint positions
            current_joint_pos = self.bus.sync_read("Present_Position")
            self.current_joint_pos = np.array([current_joint_pos[name] for name in self.bus.motors])

        # Calculate current end-effector position using forward kinematics
        if self.current_ee_pos is None:
            self.current_ee_pos = self.kinematics.forward_kinematics(self.current_joint_pos, frame=EE_FRAME)

        # Set desired end-effector position by adding delta
        desired_ee_pos = np.eye(4)
        desired_ee_pos[:3, :3] = self.current_ee_pos[:3, :3]  # Keep orientation
        
        # Add delta to position and clip to bounds
        desired_ee_pos[:3, 3] = self.current_ee_pos[:3, 3] + action[:3]
        print(desired_ee_pos)
        if self.end_effector_bounds is not None:
            desired_ee_pos[:3, 3] = np.clip(
                desired_ee_pos[:3, 3],
                self.end_effector_bounds["min"],
                self.end_effector_bounds["max"],
            )

        # Compute inverse kinematics to get joint positions
        target_joint_values_in_degrees = self.kinematics.ik(
            self.current_joint_pos, desired_ee_pos, position_only=True, frame=EE_FRAME
        )

        target_joint_values_in_degrees = np.clip(target_joint_values_in_degrees, -180.0, 180.0)
        # Create joint space action dictionary
        joint_action = {
            f"{key}.pos": target_joint_values_in_degrees[i] for i, key in enumerate(self.bus.motors.keys())
        }

        # Handle gripper separately if included in action
        # Gripper delta action is in the range 0 - 2,
        # We need to shift the action to the range -1, 1 so that we can expand it to -Max_gripper_pos, Max_gripper_pos
        joint_action["gripper.pos"] = np.clip(
            self.current_joint_pos[-1] + (action[-1] - 1) * self.config.max_gripper_pos,
            5,
            self.config.max_gripper_pos,
        )

        self.current_ee_pos = desired_ee_pos.copy()
        self.current_joint_pos = target_joint_values_in_degrees.copy()
        self.current_joint_pos[-1] = joint_action["gripper.pos"]

        # Send joint space action to parent class
        return super().send_action(joint_action)

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()
        obs_dict = self.bus.sync_read("Present_Position")
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def reset(self):
        self.current_ee_pos = None
        self.current_joint_pos = None
