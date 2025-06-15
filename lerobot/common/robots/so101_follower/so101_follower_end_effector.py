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
import threading
from typing import Any
import pyrealsense2 as rs
import mediapipe as mp
import cv2
import datetime as dt
import pyrealsense2 as rs
import mediapipe as mp
import cv2
import numpy as np
import datetime as dt

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
        self._hand_action = None
        self._hand_tracking_thread = None
        self._hand_tracking_running = False
        self._hand_action_lock = threading.Lock()
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
        
    def _hand_tracking_loop(self):
        # ...copy your hand tracking code here, but replace self.send_action(action) with:
        # with self._hand_action_lock:
        #     self._hand_action = action
        # And use self._hand_tracking_running as the loop condition.
        

        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (20, 100)
        fontScale = .5
        color = (0, 150, 255)
        thickness = 1

        realsense_ctx = rs.context()
        connected_devices = []
        for i in range(len(realsense_ctx.devices)):
            detected_camera = realsense_ctx.devices[i].get_info(rs.camera_info.serial_number)
            print(f"{detected_camera}")
            connected_devices.append(detected_camera)
        device = connected_devices[0]
        pipeline = rs.pipeline()
        config = rs.config()
        background_removed_color = 153  # Grey

        mpHands = mp.solutions.hands
        hands = mpHands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
        mpDraw = mp.solutions.drawing_utils

        config.enable_device(device)
        stream_res_x = 640
        stream_res_y = 480
        stream_fps = 30
        config.enable_stream(rs.stream.depth, stream_res_x, stream_res_y, rs.format.z16, stream_fps)
        config.enable_stream(rs.stream.color, stream_res_x, stream_res_y, rs.format.bgr8, stream_fps)
        profile = pipeline.start(config)
        align_to = rs.stream.color
        align = rs.align(align_to)

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        clipping_distance_in_meters = 2
        clipping_distance = clipping_distance_in_meters / depth_scale

        def get_3d_point(lm, depth_image, width, height, depth_scale):
            x = int(lm.x * width)
            y = int(lm.y * height)
            if x >= width: x = width - 1
            if y >= height: y = height - 1
            depth = depth_image[y, x] * depth_scale
            return np.array([x, y, depth])

        while self._hand_tracking_running:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            depth_image_flipped = cv2.flip(depth_image, 1)
            color_image = np.asanyarray(color_frame.get_data())
            color_image = cv2.flip(color_image, 1)
            images = np.where(
                (np.dstack((depth_image_flipped,)*3) > clipping_distance) | (np.dstack((depth_image_flipped,)*3) <= 0),
                background_removed_color,
                color_image
            )
            color_images_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            results = hands.process(color_images_rgb)

            if results.multi_hand_landmarks and results.multi_handedness:
                for i, (handLms, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                    hand_label = handedness.classification[0].label
                    width, height = color_image.shape[1], color_image.shape[0]
                    action = {
                        "delta_x": 0.0,
                        "delta_y": 0.0,
                        "delta_z": 0.0,
                        "gripper": 0.0,
                        "wrist_roll": 0.0,  # Placeholder for wrist roll
                    }
                    if hand_label == "Right":
                        thumb_tip = handLms.landmark[4]
                        finger_tips = [handLms.landmark[8], handLms.landmark[12], handLms.landmark[16], handLms.landmark[20]]
                        finger_names = ["Index", "Middle", "Ring", "Pinky"]

                        thumb_vec = np.array([thumb_tip.x, thumb_tip.y, thumb_tip.z])
                        angles = []
                        for tip, name in zip(finger_tips, finger_names):
                            finger_vec = np.array([tip.x, tip.y, tip.z])
                            v1 = thumb_vec
                            v2 = finger_vec
                            dot = np.dot(v1, v2)
                            norm1 = np.linalg.norm(v1)
                            norm2 = np.linalg.norm(v2)
                            if norm1 > 0 and norm2 > 0:
                                angle_rad = np.arccos(np.clip(dot / (norm1 * norm2), -1.0, 1.0))
                                angle_deg = np.degrees(angle_rad)
                                angles.append((name, angle_deg))
                            else:
                                angles.append((name, None))
                        middle_knuckle = handLms.landmark[9]
                        hand_3d = get_3d_point(middle_knuckle, depth_image_flipped, width, height, depth_scale)
                        cam_center = np.array([width // 2, height // 2, depth_image_flipped[height // 2, width // 2] * depth_scale])
                        axis_x = hand_3d[0] - cam_center[0]
                        axis_y = (hand_3d[2] / depth_scale) - (cam_center[2] / depth_scale)
                        axis_z = hand_3d[1] - cam_center[1]
                        action["gripper"] = angles[0][1] if angles and angles[0][1] is not None else 0.0
                        
                    if hand_label== "Left":
                        wrist = handLms.landmark[0]
                        index_base = handLms.landmark[5]
                        pinky_base = handLms.landmark[17]
                        # Vector from wrist to index base and wrist to pinky base
                        v1 = np.array([index_base.x - wrist.x, index_base.y - wrist.y])
                        v2 = np.array([pinky_base.x - wrist.x, pinky_base.y - wrist.y])
                        # Angle between v1 and horizontal axis (x axis)
                        angle_rad = np.arctan2(v1[1], v1[0])
                        angle_deg = np.degrees(angle_rad)
                        action["wrist_roll"] =  angle_deg
                        
                    with self._hand_action_lock:
                        self._hand_action = action
            # Optionally show the window for debug
            cv2.imshow("Hand Tracking", images)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        pipeline.stop()
        print("Hand tracking stopped.")
    
    def start_hand_tracking(self):
        """Start hand tracking in a background thread."""
        if self._hand_tracking_thread is None or not self._hand_tracking_thread.is_alive():
            self._hand_tracking_running = True
            self._hand_tracking_thread = threading.Thread(target=self._hand_tracking_loop, daemon=True)
            self._hand_tracking_thread.start()
            
    def get_hand_action(self):
        """Get the latest hand action (thread-safe)."""
        with self._hand_action_lock:
            return self._hand_action.copy() if self._hand_action else None

    def stop_hand_tracking(self):
        """Stop the hand tracking thread."""
        self._hand_tracking_running = False
        if self._hand_tracking_thread is not None:
            self._hand_tracking_thread.join(timeout=2)
            self._hand_tracking_thread = None
        
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
        print("JOINT KEYS:", joint_keys)
        if all(k in action for k in joint_keys):
            self.start_hand_tracking()
            
            shadow_action = self.get_hand_action()
            print("Hand action received:", shadow_action)
            
            
            # Optionally: update current_joint_pos and current_ee_pos
            self.current_joint_pos = np.array([action[k] for k in joint_keys])
            self.current_ee_pos = self.kinematics.forward_kinematics(self.current_joint_pos, frame=EE_FRAME)
            if shadow_action:
                gripper_angle = shadow_action["gripper"]
                min_angle_1 = 5.0   
                max_angle_1 = 20.0  
                gripper_min_1 = 5  
                gripper_max_1 = 70  
                gripper_angle = np.clip(gripper_angle, min_angle_1, max_angle_1)
                gripper_mapped = gripper_min_1 + (gripper_angle - min_angle_1) * (gripper_max_1 - gripper_min_1) / (max_angle_1 - min_angle_1)
                gripper_mapped = np.clip(gripper_mapped, gripper_min_1, gripper_max_1)
                action['gripper.pos'] = gripper_mapped
                print("Action after changing")
                print(action)
                min_angle = -140.0  
                max_angle = -10.0  
                gripper_min = 10  
                gripper_max = 90 
                wrist_roll_angle = shadow_action["wrist_roll"]
                wrist_roll_angle = np.clip(wrist_roll_angle, min_angle, max_angle)
                wrist_roll_mapped = gripper_min + abs(wrist_roll_angle - min_angle) * (gripper_max - gripper_min) / (max_angle - min_angle)
                wrist_roll_mapped = np.clip(wrist_roll_mapped, gripper_min, gripper_max)
                action['wrist_roll.pos'] = wrist_roll_angle
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
        print("Desired end-effector position:")
        print(desired_ee_pos)
        print(type(desired_ee_pos))
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
        # return super().send_action(joint_action)

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
