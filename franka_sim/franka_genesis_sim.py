import argparse
import logging
import os
import platform
import sys
import threading
import time
from enum import Enum
from pathlib import Path

import genesis as gs
import numpy as np

from typing import Callable

from .scenes.force_test_scene import init_scene

# import pinocchio as pin

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SceneInitFunc = Callable[[gs.Scene], None]

class ControlMode(Enum):
    POSITION = "position"
    VELOCITY = "velocity"
    TORQUE = "torque"
    NONE = "none"


class FrankaGenesisSim:
    def __init__(self, enable_vis : bool = False):
        self.enable_vis = enable_vis
        self.scene = None
        self.franka_left = None
        self.franka_right = None
        self.model = None
        self.data = None
        self.running = False
        
        # Separate state tracking for left and right robots
        self.latest_torques_left = np.zeros(7)
        self.latest_joint_positions_left = np.zeros(7)
        self.latest_joint_velocities_left = np.zeros(7)
        self.latest_torques_right = np.zeros(7)
        self.latest_joint_positions_right = np.zeros(7)
        self.latest_joint_velocities_right = np.zeros(7)
        
        # Backward compatibility - use left robot as default
        self.latest_torques = self.latest_torques_left
        self.latest_joint_positions = self.latest_joint_positions_left
        self.latest_joint_velocities = self.latest_joint_velocities_left
        
        self.torque_lock = threading.Lock()
        self.joint_position_lock = threading.Lock()
        self.joint_velocity_lock = threading.Lock()
        self.control_mode = ControlMode.POSITION  # Default to position control
        self.control_mode_lock = threading.Lock()
        self.dt = 0.01  # Simulation timestep
        self.sim_thread = None
        self.ddq_filtered = np.zeros(9)

        # Get the Genesis assets path instead of our own
        import genesis

        genesis_path = Path(genesis.__file__).parent
        self.xml_path = genesis_path / "assets/xml/franka_emika_panda/panda.xml"

        # Keep URDF path for future use if needed (for Pinocchio)
        # This is currently unused, but kept for reference
        current_dir = Path(__file__).parent
        assets_dir = current_dir.parent / "assets"
        self.urdf_path = assets_dir / "urdf/panda_bullet/panda.urdf"

        logger.info(f"Using Genesis XML path: {self.xml_path}")

    def load_panda_model(self):
        pass
        # TODO: load pinocchio model
        # model = pin.buildModelFromUrdf(str(self.urdf_path))
        # data = model.createData()
        # return model, data

    def initialize_simulation(self):
        # Initialize Genesis with CPU backend
        gs.init(backend=gs.cpu, logging_level=None)

        # Create scene
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0, -3.5, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                res=(1280, 800),
                max_FPS=60,
            ),
            sim_options=gs.options.SimOptions(
                dt=self.dt,
            ),
            show_viewer=self.enable_vis,
            show_FPS=False,
        )

        # Add entities
        self.scene.add_entity(gs.morphs.Plane())

        init_scene(scene=self.scene)

        self.base_frame_pos_left = np.array([-0.375,  -0.281, -0.025])
        self.base_frame_pos_right = np.array([-0.375,  0.281, -0.025])

        self.franka_left = self.scene.add_entity(
            gs.morphs.MJCF(
                file=str(self.xml_path),
                pos = self.base_frame_pos_left
            ),
            material=gs.materials.Rigid(gravity_compensation=1.0),
        )


        self.franka_right = self.scene.add_entity(
            gs.morphs.MJCF(
                file=str(self.xml_path),
                pos = self.base_frame_pos_right
            ),
            material=gs.materials.Rigid(gravity_compensation=1.0),
        )

        # Build scene
        self.scene.build()

        # Joint names and indices
        self.jnt_names_left = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
            "finger_joint1",
            "finger_joint2",
        ]
        self.dofs_idx_left = [self.franka_left.get_joint(name).dofs_idx_local[0] for name in self.jnt_names_left]

        # Set force range for safety
        self.franka_left.set_dofs_force_range(
            lower=np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            upper=np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
            dofs_idx_local=self.dofs_idx_left,
        )

        # Initialize to default position for left robot
        initial_q_left = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.785])
        # Set the initial position as the target position for the controller
        with self.joint_position_lock:
            self.latest_joint_positions_left = initial_q_left.copy()
            # Update the reference to point to left robot
            self.latest_joint_positions = self.latest_joint_positions_left

        # Joint names and indices
        self.jnt_names_right = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
            "finger_joint1",
            "finger_joint2",
        ]
        self.dofs_idx_right = [self.franka_right.get_joint(name).dofs_idx_local[0] for name in self.jnt_names_right]

        # Set force range for safety
        self.franka_right.set_dofs_force_range(
            lower=np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            upper=np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
            dofs_idx_local=self.dofs_idx_right,
        )

        # Initialize to default position for right robot
        initial_q_right = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.785])
        # Set the initial position for right robot (separate from left)
        self.latest_joint_positions_right = initial_q_right.copy()

        for _ in range(100):
            self.franka_left.set_dofs_position(np.concatenate([initial_q_left, [0.04, 0.04]]), self.dofs_idx_left)
            self.franka_right.set_dofs_position(np.concatenate([initial_q_right, [0.04, 0.04]]), self.dofs_idx_right)
            # logger.info(f"DOFs position: {self.franka_left.get_dofs_position(self.dofs_idx_left).cpu().numpy()}")
            # hand_link = self.franka_left.get_link("hand")
            # ee_pos = hand_link.get_pos().cpu().numpy()
            # ee_quat = hand_link.get_quat().cpu().numpy()
            # logger.info(f"End effector position: {ee_pos}, orientation: {ee_quat}")
            self.scene.step()

    def set_control_mode(self, mode: ControlMode):
        """Set the control mode for the robot"""
        if not isinstance(mode, ControlMode):
            raise ValueError(f"Mode must be a ControlMode enum, got {type(mode)}")

        with self.control_mode_lock:
            logger.info(f"Switching control mode to: {mode.value}")
            self.control_mode = mode

    def update_torques(self, torques):
        """Update the latest torques to be applied in simulation"""
        with self.torque_lock:
            self.latest_torques = np.array(torques)

    def update_joint_positions(self, positions):
        """Update the latest joint positions to be applied in simulation"""
        with self.joint_position_lock:
            self.latest_joint_positions = np.array(positions)

    def update_joint_velocities(self, velocities):
        """Update the latest joint velocities to be applied in simulation"""
        with self.joint_velocity_lock:
            self.latest_joint_velocities = np.array(velocities)

    def run_simulation(self):
        """Main simulation loop"""
        logger.info("Starting Genesis simulation loop")


        # For numerical differentiation
        self.prev_dq_full = np.zeros(9)
        self.ddq_filtered = np.zeros(9)
        alpha_acc = 0.95

        while self.running:
            # Get current joint states
            q_full = self.franka_left.get_dofs_position(self.dofs_idx_left).cpu().numpy()
            dq_full = self.franka_left.get_dofs_velocity(self.dofs_idx_left).cpu().numpy()

            # Calculate acceleration
            ddq_raw = (dq_full - self.prev_dq_full) / self.dt
            self.ddq_filtered = alpha_acc * self.ddq_filtered + (1 - alpha_acc) * ddq_raw
            self.prev_dq_full = dq_full.copy()

            # Get current control mode
            with self.control_mode_lock:
                current_mode = self.control_mode

            # Apply control based on mode
            if current_mode == ControlMode.POSITION:
                with self.joint_position_lock:
                    q_d = self.latest_joint_positions.copy()
                q_cmd = np.concatenate([q_d, [0.04, 0.04]])
                self.franka_left.control_dofs_position(q_cmd, self.dofs_idx_left)

            elif current_mode == ControlMode.VELOCITY:
                with self.joint_velocity_lock:
                    dq_d = self.latest_joint_velocities.copy()
                dq_cmd = np.concatenate([dq_d, [0.0, 0.0]])
                self.franka_left.control_dofs_velocity(dq_cmd, self.dofs_idx_left)

            elif current_mode == ControlMode.TORQUE:
                with self.torque_lock:
                    tau_d = self.latest_torques.copy()
                tau_cmd = np.concatenate([tau_d, [0.0, 0.0]])
                self.franka_left.control_dofs_force(tau_cmd, self.dofs_idx_left)

            # Step simulation
            self.scene.step()

            # Optional: Add small sleep to prevent too high CPU usage
            time.sleep(0.001)


            

    def start(self):
        """Start the simulation"""
        if not self.scene:
            self.initialize_simulation()

        self.running = True

        if self.enable_vis:
            # Run simulation in a separate thread when visualization is enabled
            # if macos, run in a separate thread
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                gs.tools.run_in_another_thread(fn=self.run_simulation, args=())
            else:
                self.run_simulation()
            # Run viewer in main thread
            # self.scene.viewer.start()
        else:
            # Without visualization, just run simulation in current thread
            self.run_simulation()

    def stop(self):
        """Stop the simulation"""
        self.running = False
        self.scene.stop_recording()
        if self.enable_vis:
            self.scene.viewer.stop()
        if self.sim_thread:
            self.sim_thread.join(timeout=1.0)  # Wait for simulation thread to finish

    def get_robot_state(self):
        """Get current robot state for network transmission"""
        # q_d is the desired joint positions user sent joint positions
        q_d = self.latest_joint_positions

        q_full = self.franka_left.get_dofs_position(self.dofs_idx_left).cpu().numpy()
        dq_full = self.franka_left.get_dofs_velocity(self.dofs_idx_left).cpu().numpy()
        # calculate ddq_full
        ddq_full = self.ddq_filtered

        # Get end-effector position and orientation
        hand_link = self.franka_left.get_link("hand")
        ee_pos_world = hand_link.get_pos().cpu().numpy()
        ee_quat = hand_link.get_quat().cpu().numpy()  # [x, y, z, w]

        # Convert end-effector position from world frame to robot base frame
        ee_pos = ee_pos_world - self.base_frame_pos_left
        
        logger.debug(f"ee_pos_world: {ee_pos_world}, base_pos: {self.base_frame_pos_left}, ee_pos_relative: {ee_pos}")
        
        # Convert quaternion to rotation matrix
        # Note: quaternion from Genesis is [x, y, z, w]
        x, y, z, w = ee_quat
        R = np.array(
            [
                [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
            ]
        )

        # Construct homogeneous transformation matrix with corrected position
        O_T_EE = np.eye(4)
        O_T_EE[:3, :3] = R
        O_T_EE[:3, 3] = ee_pos  # Use the corrected position relative to base

        # Convert to column-major 16-element array
        O_T_EE = O_T_EE.T.flatten()

        # Return only the first 7 joints (excluding fingers)
        return {
            "q": q_full[:7],
            "dq": dq_full[:7],
            "ddq": ddq_full[:7],
            "q_d": q_d,
            "dq_d": dq_full[:7],
            "ddq_d": ddq_full[:7],
            "tau_J": self.latest_torques,  # Current commanded torques
            "O_T_EE": O_T_EE,  # End-effector pose in base frame (column-major)
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    sim = FrankaGenesisSim(enable_vis=args.vis)
    sim.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        sim.stop()


if __name__ == "__main__":
    main()
