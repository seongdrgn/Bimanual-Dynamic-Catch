import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg, mdp
from omni.isaac.lab.envs import DirectMARLEnv, DirectMARLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg, PhysxCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import sample_uniform
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.markers import VisualizationMarkersCfg, VisualizationMarkers
from omni.isaac.lab.sensors import CameraCfg, Camera, ContactSensorCfg, ContactSensor

from omni.isaac.lab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import os
import torch
from typing import Sequence, Dict
import math
import numpy as np

##
# Pre-defined configs
##
from torch._tensor import Tensor
from .bimanual_catch_mappo_cfg import BimanualCatchEnvCfgV1

class BimanualCatchEnvV1(DirectMARLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: BimanualCatchEnvCfgV1

    def __init__(self, cfg: BimanualCatchEnvCfgV1,
                 render_mode: str | None = None, **kwargs):
        
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self.num_R_robot_dofs = self.R_robot.num_joints
        self.num_L_robot_dofs = self.L_robot.num_joints

        R_joint_pos_limits = self.R_robot.root_physx_view.get_dof_limits().to(self.device)
        self.R_robot_dof_lower_limits = R_joint_pos_limits[..., 0]
        self.R_robot_dof_upper_limits = R_joint_pos_limits[..., 1]
        self.R_robot_dof_speed_scales = torch.ones_like(self.R_robot_dof_lower_limits)
        self.R_palm_link_idx = self.R_robot.find_bodies("palm_link")[0][0]

        L_joint_pos_limits = self.L_robot.root_physx_view.get_dof_limits().to(self.device)
        self.L_robot_dof_lower_limits = L_joint_pos_limits[..., 0]
        self.L_robot_dof_upper_limits = L_joint_pos_limits[..., 1]
        self.L_robot_dof_speed_scales = torch.ones_like(self.L_robot_dof_lower_limits)
        self.L_palm_link_idx = self.L_robot.find_bodies("palm_link")[0][0]

        # list of actuated joints
        self.R_robot_dof_indices = list()
        for joint_name in self.cfg.actuated_joint_names:
            self.R_robot_dof_indices.append(self.R_robot.joint_names.index(joint_name))
        
        self.L_robot_dof_indices = list()
        for joint_name in self.cfg.actuated_joint_names:
            self.L_robot_dof_indices.append(self.L_robot.joint_names.index(joint_name))

        # buffers for position targets
        self.R_robot_dof_targets = torch.zeros((self.num_envs, self.num_R_robot_dofs), dtype=torch.float, device=self.device)
        self.R_prev_targets = torch.zeros((self.num_envs, self.num_R_robot_dofs), dtype=torch.float, device=self.device)
        self.R_cur_targets = torch.zeros((self.num_envs, self.num_R_robot_dofs), dtype=torch.float, device=self.device)
        self.L_robot_dof_targets = torch.zeros((self.num_envs, self.num_L_robot_dofs), dtype=torch.float, device=self.device)
        self.L_prev_targets = torch.zeros((self.num_envs, self.num_L_robot_dofs), dtype=torch.float, device=self.device)
        self.L_cur_targets = torch.zeros((self.num_envs, self.num_L_robot_dofs), dtype=torch.float, device=self.device)

        # unit tensors
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        # track successes
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # buffers for states
        self.states_stack_frames = []
        for i in range(self.cfg.num_stacks):
            self.states_stack_frames.append(torch.zeros((self.num_envs, self.cfg.one_frame_states), device=self.device))

        # buffers for obs
        self.R_robot_obs_buf_stack_frames = []
        for i in range(self.cfg.num_stacks):
            self.R_robot_obs_buf_stack_frames.append(torch.zeros((self.num_envs, self.cfg.one_frame_obs), device=self.device))
        self.R_robot_obs_buf = torch.zeros((self.num_envs, self.cfg.one_frame_obs*self.cfg.num_stacks), device=self.device)

        self.L_robot_obs_buf_stack_frames = []
        for i in range(self.cfg.num_stacks):
            self.L_robot_obs_buf_stack_frames.append(torch.zeros((self.num_envs, self.cfg.one_frame_obs), device=self.device))
        self.L_robot_obs_buf = torch.zeros((self.num_envs, self.cfg.one_frame_obs*self.cfg.num_stacks), device=self.device)

        self.R_closest_dist = torch.full((self.num_envs,), float("inf"), device=self.device)
        self.L_closest_dist = torch.full((self.num_envs,), float("inf"), device=self.device)

    def _setup_scene(self):
        self.R_robot = Articulation(self.cfg.R_robot)
        self.L_robot = Articulation(self.cfg.L_robot)
        self._object = RigidObject(self.cfg.cylinder_cfg)
        self.R_contact_sensors = ContactSensor(self.cfg.R_contact_sensors)
        self.L_contact_sensors = ContactSensor(self.cfg.L_contact_sensors)
        self._table = RigidObject(self.cfg.table)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        self.scene.articulations["R_robot"] = self.R_robot
        self.scene.articulations["L_robot"] = self.L_robot
        self.scene.rigid_objects["object"] = self._object
        self.scene.sensors["R_contact_sensors"] = self.R_contact_sensors
        self.scene.sensors["L_contact_sensors"] = self.L_contact_sensors
        # self.scene.rigid_objects["table"] = self._table

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # pre-physics step calls

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]):
        self.actions = actions

    def _apply_action(self):
        # apply right robot actions
        self.R_cur_targets[:,self.R_robot_dof_indices[6:]] = scale(
            self.actions["R_robot"][:,6:],
            self.R_robot_dof_lower_limits[:,self.R_robot_dof_indices[6:]],
            self.R_robot_dof_upper_limits[:,self.R_robot_dof_indices[6:]],
        )
        self.R_cur_targets[:,self.R_robot_dof_indices[6:]] = (
            self.cfg.act_moving_average * self.R_cur_targets[:,self.R_robot_dof_indices[6:]]
            + (1.0 - self.cfg.act_moving_average) * self.R_prev_targets[:,self.R_robot_dof_indices[6:]]
        )

        self.R_cur_targets[:,self.R_robot_dof_indices[:6]] = self.R_prev_targets[:,self.R_robot_dof_indices[:6]] + self.actions["R_robot"][:,:6] * self.cfg.action_scale * self.dt

        self.R_cur_targets[:,self.R_robot_dof_indices[:]] = saturate(
            self.R_cur_targets[:, self.R_robot_dof_indices[:]],
            self.R_robot_dof_lower_limits[:, self.R_robot_dof_indices[:]],
            self.R_robot_dof_upper_limits[:, self.R_robot_dof_indices[:]],
        )

        # apply left robot actions
        self.L_cur_targets[:,self.L_robot_dof_indices[6:]] = scale(
            self.actions["L_robot"][:,6:],
            self.L_robot_dof_lower_limits[:,self.L_robot_dof_indices[6:]],
            self.L_robot_dof_upper_limits[:,self.L_robot_dof_indices[6:]],
        )
        self.L_cur_targets[:,self.L_robot_dof_indices[6:]] = (
            self.cfg.act_moving_average * self.L_cur_targets[:,self.L_robot_dof_indices[6:]]
            + (1.0 - self.cfg.act_moving_average) * self.L_prev_targets[:,self.L_robot_dof_indices[6:]]
        )

        self.L_cur_targets[:,self.L_robot_dof_indices[:6]] = self.L_prev_targets[:,self.L_robot_dof_indices[:6]] + self.actions["L_robot"][:,:6] * self.cfg.action_scale * self.dt

        self.L_cur_targets[:,self.L_robot_dof_indices[:]] = saturate(
            self.L_cur_targets[:, self.L_robot_dof_indices[:]],
            self.L_robot_dof_lower_limits[:, self.L_robot_dof_indices[:]],
            self.L_robot_dof_upper_limits[:, self.L_robot_dof_indices[:]],
        )

        self.R_prev_targets[:,self.R_robot_dof_indices] = self.R_cur_targets[:,self.R_robot_dof_indices]
        self.L_prev_targets[:,self.L_robot_dof_indices] = self.L_cur_targets[:,self.L_robot_dof_indices]

        self.R_robot.set_joint_position_target(self.R_cur_targets[:,self.R_robot_dof_indices[:]], joint_ids=self.R_robot_dof_indices[:])
        self.L_robot.set_joint_position_target(self.L_cur_targets[:,self.L_robot_dof_indices[:]], joint_ids=self.L_robot_dof_indices[:])

    # post-physics step calls

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        drop_object = (self._object.data.body_pos_w[:,0,2] < self.cfg.object_height_threshold) & (self.episode_length_buf > 10)
        time_outs = self.episode_length_buf >= self.max_episode_length

        terminated = {agent: drop_object for agent in self.cfg.possible_agents}
        time_outs = {agent: time_outs for agent in self.cfg.possible_agents}
        return terminated, time_outs

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()

        (
            total_reward,
            self.successes[:],
            self.R_closest_dist,
            self.L_closest_dist,
        ) = compute_rewards(
            self.reset_buf,
            self.successes,
            self.episode_length_buf,
            self.max_episode_length,
            self.object_pos,
            self.R_hand_pos,
            self.L_hand_pos,
            self.R_closest_dist,
            self.L_closest_dist,
            self.R_contact_sensors_val,
            self.L_contact_sensors_val,
            self.actions,
            self.cfg.R_dist_reward_scale,
            self.cfg.L_dist_reward_scale,
            self.cfg.action_penalty_scale,
            self.cfg.collision_penalty_scale,
            self.cfg.drop_penalty,
            self.cfg.reach_goal_bonus,
            self.cfg.hit_bonus,
            self.cfg.success_tolerance,
        )
        
        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["R_robot_reward"] = total_reward["R_robot"]
        self.extras["log"]["L_robot_reward"] = total_reward["L_robot"]

        return total_reward

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self.R_robot._ALL_INDICES    

        super()._reset_idx(env_ids)
        for i in range(self.cfg.num_stacks):
            self.R_robot_obs_buf_stack_frames[i][env_ids] = torch.zeros((self.cfg.one_frame_obs), device=self.device)
        self.R_robot_obs_buf[env_ids] = torch.zeros((self.cfg.one_frame_obs*self.cfg.num_stacks),device=self.device)

        for i in range(self.cfg.num_stacks):
            self.L_robot_obs_buf_stack_frames[i][env_ids] = torch.zeros((self.cfg.one_frame_obs), device=self.device)
        self.L_robot_obs_buf[env_ids] = torch.zeros((self.cfg.one_frame_obs*self.cfg.num_stacks),device=self.device)

        # set right robot states
        R_joint_pos = self.R_robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self.R_robot.num_joints),
            self.device,
        )

        R_joint_pos = torch.clamp(R_joint_pos, self.R_robot_dof_lower_limits[env_ids], self.R_robot_dof_upper_limits[env_ids])
        R_joint_vel = torch.zeros_like(R_joint_pos)
        self.R_robot.set_joint_position_target(R_joint_pos, env_ids=env_ids)
        self.R_robot.write_joint_state_to_sim(R_joint_pos, R_joint_vel, env_ids=env_ids)
        self.R_robot_dof_targets[env_ids,:] = R_joint_pos
        self.R_prev_targets[env_ids,:] = R_joint_pos

        # set left robot states
        L_joint_pos = self.L_robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self.L_robot.num_joints),
            self.device,
        )

        L_joint_pos = torch.clamp(L_joint_pos, self.L_robot_dof_lower_limits[env_ids], self.L_robot_dof_upper_limits[env_ids])
        L_joint_vel = torch.zeros_like(L_joint_pos)
        self.L_robot.set_joint_position_target(L_joint_pos, env_ids=env_ids)
        self.L_robot.write_joint_state_to_sim(L_joint_pos, L_joint_vel, env_ids=env_ids)
        self.L_robot_dof_targets[env_ids,:] = L_joint_pos
        self.L_prev_targets[env_ids,:] = L_joint_pos

        # object state
        object_default_state = self._object.data.default_root_state.clone()[env_ids]
        random_pos, random_vel, goal_pos = self.get_object_random_pose(env_ids=env_ids)

        object_default_state[:, 0:3] = (
            object_default_state[:, 0:3] + random_pos + self.scene.env_origins[env_ids]
        )
        object_default_state[:, 7:10] = (
            object_default_state[:, 7:10] + random_vel
        )
        self._object.write_root_state_to_sim(object_default_state, env_ids)

        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self._compute_intermediate_values()

    def get_object_random_pose(self, env_ids: torch.Tensor | None):
        Xs = torch.rand_like(self._object.data.default_root_state[env_ids,0])*(self.cfg.range_Xs[1]-self.cfg.range_Xs[0]) + self.cfg.range_Xs[0]
        Ys = torch.rand_like(Xs)*(self.cfg.range_Ys[1]-self.cfg.range_Ys[0]) + self.cfg.range_Ys[0]
        Zs = torch.rand_like(Xs)*(self.cfg.range_Zs[1]-self.cfg.range_Zs[0]) + self.cfg.range_Zs[0]

        # t를 먼저 선택
        t = torch.rand_like(Xs)*(self.cfg.range_t[1]-self.cfg.range_t[0]) + self.cfg.range_t[0]

        # Xg 선택
        Xg = torch.rand_like(Xs)*(self.cfg.robot_reach)
        Xg = torch.clamp_min(Xg, 0.5)

        # Vx 계산
        Vx = (Xg-Xs)/t

        # Yg의 범위 설정
        max_Yg = torch.clamp_min((self.cfg.robot_reach*self.cfg.robot_reach - Xg * Xg)**0.5, 0.)
        # Yg를 random으로 선택
        Yg = torch.rand_like(Xs)*(max_Yg-(-max_Yg)) + (-max_Yg)

        # Vy를 계산
        Vy = (Yg - Ys) / t

        # Zg의 범위 설정
        max_Zg = torch.clamp_min((self.cfg.robot_reach*self.cfg.robot_reach - Xg * Xg - Yg * Yg)**0.5, 0.)

        # Zg를 random으로 선택
        Zg = torch.rand_like(Xs)*(max_Zg)

        # Vz를 계산
        Vz = (Zg - Zs + 0.5 * self.cfg.gravity * (t ** 2)) / t
        Zg = torch.clamp_min(Zg+self.cfg.Z_offset, self.cfg.Z_offset)
        Zs += self.cfg.Z_offset

        random_pos = torch.cat([Xs.unsqueeze(-1),Ys.unsqueeze(-1),Zs.unsqueeze(-1)],dim=-1)
        random_vel = torch.cat([Vx.unsqueeze(-1),Vy.unsqueeze(-1),Vz.unsqueeze(-1)],dim=-1)

        goal_pos = torch.cat([Xg.unsqueeze(-1),Yg.unsqueeze(-1),Zg.unsqueeze(-1)],dim=-1)

        return random_pos, random_vel, goal_pos

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self.compute_R_robot_state()
        self.compute_L_robot_state()

        observations = {
                        "R_robot": self.R_robot_obs_buf,
                        "L_robot": self.L_robot_obs_buf,
                        }        
        
        return observations

    def _get_states(self) -> torch.Tensor:
        states = torch.cat(
            (
                # ---- right robot ----
                self.R_hand_pos,
                self.R_hand_rot,
                self.R_robot_joint_pos,
                self.R_contact_sensors_val,
                self.actions["R_robot"],
                # ---- left robot ----
                self.L_hand_pos,
                self.L_hand_rot,
                self.L_robot_joint_pos,
                self.L_contact_sensors_val,
                self.actions["L_robot"],
                # ---- object ----
                self.object_pos,
            ),
            dim=-1
        )

        for i in range(len(self.states_stack_frames)-1):
            states = torch.cat((states, self.states_stack_frames[i]), dim=-1)
            self.states_stack_frames[i] = states[:, (i)*self.cfg.one_frame_states:(i+1)*self.cfg.one_frame_states].clone()

        return states

    def compute_R_robot_state(self):
        self.R_robot_obs_buf[:,0:3] = self.R_hand_pos
        self.R_robot_obs_buf[:,3:7] = self.R_hand_rot
        self.R_robot_obs_buf[:,7:25] = self.R_robot_joint_pos
        self.R_robot_obs_buf[:,25:41] = self.R_contact_sensors_val
        self.R_robot_obs_buf[:,41:59] = self.actions['R_robot']
        self.R_robot_obs_buf[:,59:62] = self.object_pos
        self.R_robot_obs_buf[:,62:65] = self.L_hand_pos

        for i in range(len(self.R_robot_obs_buf_stack_frames)-1):
            self.R_robot_obs_buf[:, (i+1)*self.cfg.one_frame_obs:(i+2)*self.cfg.one_frame_obs] = self.R_robot_obs_buf_stack_frames[i]
            self.R_robot_obs_buf_stack_frames[i] = self.R_robot_obs_buf[:, (i)*self.cfg.one_frame_obs:(i+1)*self.cfg.one_frame_obs].clone()

    def compute_L_robot_state(self):
        self.L_robot_obs_buf[:,0:3] = self.L_hand_pos
        self.L_robot_obs_buf[:,3:7] = self.L_hand_rot
        self.L_robot_obs_buf[:,7:25] = self.L_robot_joint_pos
        self.L_robot_obs_buf[:,25:41] = self.L_contact_sensors_val
        self.L_robot_obs_buf[:,41:59] = self.actions['L_robot']
        self.L_robot_obs_buf[:,59:62] = self.object_pos
        self.L_robot_obs_buf[:,62:65] = self.R_hand_pos

        for i in range(len(self.L_robot_obs_buf_stack_frames)-1):
            self.L_robot_obs_buf[:, (i+1)*self.cfg.one_frame_obs:(i+2)*self.cfg.one_frame_obs] = self.L_robot_obs_buf_stack_frames[i]
            self.L_robot_obs_buf_stack_frames[i] = self.L_robot_obs_buf[:, (i)*self.cfg.one_frame_obs:(i+1)*self.cfg.one_frame_obs].clone()

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self.R_robot._ALL_INDICES

        # get right robot states
        self.R_hand_pos = self.R_robot.data.body_pos_w[env_ids, self.R_palm_link_idx]
        self.R_hand_pos -= self.scene.env_origins
        self.R_hand_rot = self.R_robot.data.body_quat_w[env_ids, self.R_palm_link_idx]
        self.R_robot_joint_pos = self.R_robot.data.joint_pos
        R_contact_sensors_val_raw = self.scene["R_contact_sensors"].data.net_forces_w[..., 2]
        self.R_contact_sensors_val = torch.where(R_contact_sensors_val_raw > 0.01, 1.0, 0.0)
        # print("Right Contact",R_contact_sensors_val_raw)

        # get left robot states
        self.L_hand_pos = self.L_robot.data.body_pos_w[env_ids, self.L_palm_link_idx]
        self.L_hand_pos -= self.scene.env_origins
        self.L_hand_rot = self.L_robot.data.body_quat_w[env_ids, self.L_palm_link_idx]
        self.L_robot_joint_pos = self.L_robot.data.joint_pos
        L_contact_sensors_val_raw = self.scene["L_contact_sensors"].data.net_forces_w[..., 2]
        self.L_contact_sensors_val = torch.where(L_contact_sensors_val_raw > 0.01, 1.0, 0.0)
        # print("Left Contact",L_contact_sensors_val_raw)
        
        # get object position
        self.object_pos = self._object.data.root_pos_w - self.scene.env_origins

@torch.jit.script
def compute_rewards(
    reset_buf: Tensor,
    successes: Tensor,
    episode_length: Tensor,
    max_episode_length: int,
    object_pos: Tensor,
    R_hand_pos: Tensor,
    L_hand_pos: Tensor,
    R_closest_dist: Tensor,
    L_closest_dist: Tensor,
    R_contact_sensors: Tensor,
    L_contact_sensors: Tensor,
    actions: dict[str, Tensor],
    R_dist_reward_scales: float,
    L_dist_reward_scales: float,
    action_penalty_scales: float,
    collision_penalty_scales: float,
    drop_penalty_score: float,
    reach_goal_bonus: float,
    hit_bonus: float,
    success_tolerance: float,
):

    # distance from right hand to object position
    R_object_dist = torch.norm(R_hand_pos - object_pos, p=2, dim=-1)
    R_dist_reward = torch.exp(-5.0 * R_object_dist)

    # distance from right hand to object position
    L_object_dist = torch.norm(L_hand_pos - object_pos, p=2, dim=-1)
    L_dist_reward = torch.exp(-5.0 * L_object_dist)

    # torque penalty for preventing weird motion
    R_robot_action_penalty = torch.sum(actions["R_robot"]**2, dim=-1)
    L_robot_action_penalty = torch.sum(actions["L_robot"]**2, dim=-1)

    # penalty for collision between right hand and left hand
    d_collision = torch.norm(R_hand_pos - L_hand_pos, p=2, dim=-1)
    collision_penalty = torch.exp(-5.0*d_collision)
    
    # bonus reward for hitting hand
    R_is_contact = torch.any(R_contact_sensors, dim=-1)
    L_is_contact = torch.any(L_contact_sensors, dim=-1)
    R_robot_hit_bonus_reward = torch.where(R_is_contact, hit_bonus, torch.tensor(0.0, device=R_closest_dist.device))
    L_robot_hit_bonus_reward = torch.where(L_is_contact, hit_bonus, torch.tensor(0.0, device=L_closest_dist.device))

    # penalty for dropping object
    drop_penalty = torch.where((episode_length > 10)&(object_pos[:,2] < 1.0), drop_penalty_score, torch.tensor(0.0, device=reset_buf.device))

    R_robot_reward = (
        R_dist_reward_scales * R_dist_reward
        - action_penalty_scales * R_robot_action_penalty
        - collision_penalty_scales * collision_penalty
        # + R_robot_hit_bonus_reward
        - drop_penalty
    )

    L_robot_reward = (
        L_dist_reward_scales * L_dist_reward
        - action_penalty_scales * L_robot_action_penalty
        - collision_penalty_scales * collision_penalty
        # + L_robot_hit_bonus_reward
        - drop_penalty
    )

    total_reward = {
        "R_robot": R_robot_reward,
        "L_robot": L_robot_reward
    }
    print(total_reward)
    return total_reward, successes, R_closest_dist, L_closest_dist   

@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower

@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )