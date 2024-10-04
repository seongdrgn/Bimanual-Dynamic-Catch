import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.envs import DirectMARLEnv, DirectMARLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg, PhysxCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import sample_uniform
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.sensors import CameraCfg, Camera, ContactSensorCfg, ContactSensor

from omni.isaac.lab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate
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

@configclass
class BimanualCatchEnvCfgV1(DirectMARLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # obs stack frames
    num_stacks = 2
    possible_agents = ["R_robot","L_robot"]
    action_scale = 2.0
    one_frame_obs = 65
    one_frame_states = 121
    num_states = one_frame_states * num_stacks
    num_actions: dict[str, int] = {"R_robot": 18, "L_robot": 18}
    num_observations: dict[str, int] = {"R_robot": one_frame_obs*num_stacks, "L_robot":one_frame_obs*num_stacks}

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1/120,
        render_interval=decimation,
        disable_contact_processing=False,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
        )
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048, env_spacing=6.0, replicate_physics=True)

    R_robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Right_Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/kimsy/RL-kimsy/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/bimanual_catch/assets/bimanual_right/bimanual_right.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -(2/5)*math.pi,
                "elbow_joint": -(4/6)*math.pi,
                "wrist_1_joint": -0.8*math.pi,
                "wrist_2_joint": -0.5*math.pi,
                "wrist_3_joint": 0.0,

                "joint_index_1": 0.0,
                "joint_index_2": 0.0,
                "joint_index_3": 0.0,
                "joint_middle_1": 0.0,
                "joint_middle_2": 0.0,
                "joint_middle_3": 0.0,
                "joint_ring_1":0.0,
                "joint_ring_2":0.0,
                "joint_ring_3":0.0,
                "joint_thumb_0":0.363,
                "joint_thumb_2":0.0,
                "joint_thumb_3":0.0,
            },
            pos=(0.0,-0.315,0.81),
            rot=(0.0,0.0,0.0,1.0),
        ),
        actuators={            
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_pan_joint","shoulder_lift_joint","elbow_joint","wrist_1_joint","wrist_2_joint","wrist_3_joint"],
                velocity_limit=100.0,
                effort_limit=87.0,
                stiffness=800.0,
                damping=40.0,
            ),
            "hand": ImplicitActuatorCfg(
                joint_names_expr=["joint_index_[1-3]","joint_middle_[1-3]","joint_ring_[1-3]","joint_thumb_0","joint_thumb_2","joint_thumb_3"],
                effort_limit=0.5,
                velocity_limit=100.0,
                stiffness=3.0,
                damping=0.1,
                friction=0.01,
            ),
        },
    )

    L_robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Left_Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/kimsy/RL-kimsy/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/bimanual_catch/assets/bimanual_left/bimanual_left.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "shoulder_pan_joint": math.pi,
                "shoulder_lift_joint": (2/5)*math.pi + math.pi,
                "elbow_joint": (4/6)*math.pi,
                "wrist_1_joint": 0.8*math.pi - math.pi,
                "wrist_2_joint": 0.5*math.pi,
                "wrist_3_joint": 0.0,

                "joint_index_1": 0.0,
                "joint_index_2": 0.0,
                "joint_index_3": 0.0,
                "joint_middle_1": 0.0,
                "joint_middle_2": 0.0,
                "joint_middle_3": 0.0,
                "joint_ring_1":0.0,
                "joint_ring_2":0.0,
                "joint_ring_3":0.0,
                "joint_thumb_0":0.363,
                "joint_thumb_2":0.0,
                "joint_thumb_3":0.0,
            },
            pos=(0.0,0.315,0.81),
            rot=(0.0,0.0,0.0,1.0),
        ),
        actuators={            
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_pan_joint","shoulder_lift_joint","elbow_joint","wrist_1_joint","wrist_2_joint","wrist_3_joint"],
                velocity_limit=100.0,
                effort_limit=87.0,
                stiffness=800.0,
                damping=40.0,
            ),
            "hand": ImplicitActuatorCfg(
                joint_names_expr=["joint_index_[1-3]","joint_middle_[1-3]","joint_ring_[1-3]","joint_thumb_0","joint_thumb_2","joint_thumb_3"],
                effort_limit=0.5,
                velocity_limit=100.0,
                stiffness=3.0,
                damping=0.1,
                friction=0.01,
            ),
        },
    )

    actuated_joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",

        "joint_index_1",
        "joint_index_2",
        "joint_index_3",

        "joint_middle_1",
        "joint_middle_2",
        "joint_middle_3",

        "joint_ring_1",
        "joint_ring_2",
        "joint_ring_3",
        "joint_thumb_0",

        "joint_thumb_2",
        "joint_thumb_3",
    ]

    # contact sensors
    R_contact_sensors = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Right_Robot/fsr_.*", update_period=0.0, history_length=1, debug_vis=True
    )

    # contact sensors
    L_contact_sensors = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Left_Robot/fsr_.*", update_period=0.0, history_length=1, debug_vis=True
    )

    # table
    table = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd",
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.81),
                                                  rot=(0.0, 0.0, 0.0, -1.0))
    )

    cylinder_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cylinder",
        spawn=sim_utils.CylinderCfg(
            radius=0.03,
            height=1,
            axis='Z',
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0,1.0,0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0,0.0,0.0),
            rot=(0, 0, 0.7071068, 0.7071068),
        ),
    )

    cone_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cone",
        spawn=sim_utils.ConeCfg(
            radius=0.2,
            height=0.3,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0,1.0,0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0,0.0,0.0),
            rot=(0.0,0.0,0.0,1.0),
        ),
    )

    cube_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cube",
        spawn = sim_utils.CuboidCfg(
            size=(0.15,0.15,0.15),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0,1.0,0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0,0.0,0.0),
            rot=(0.0,0.0,0.0,1.0),
        )
    )

    sphere_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Sphere",
        spawn = sim_utils.SphereCfg(
            radius=0.2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0,1.0,0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0,0.0,0.0),
            rot=(0.0,0.0,0.0,1.0),
        )
    )

    capsule_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Capsule",
        spawn = sim_utils.CapsuleCfg(
            radius=0.1,
            height=0.2,
            axis='Z',
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0,1.0,0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0,0.0,0.0),
            rot=(0.0,0.0,0.0,1.0),
        )
    )

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
    )

    # reward scales
    R_dist_reward_scale = 1.0
    L_dist_reward_scale = 1.0
    action_penalty_scale = 0.001
    collision_penalty_scale = 0.1
    hit_bonus = 3.0
    drop_penalty = 10.0
    object_height_threshold = 1.0
    reach_goal_bonus = 10.0
    success_tolerance = 0.05
    act_moving_average = 1.0

    # object initial pos
    gravity = 9.81
    X_offset = 0.
    Y_offset = 0.
    Z_offset = 0.9

    range_Xs = (2.5, 3)
    range_Ys = (-0.8, 0.8)
    range_Zs = (-0.1, 0.0)
    range_t = (0.8, 1.2)

    robot_reach = 0.8