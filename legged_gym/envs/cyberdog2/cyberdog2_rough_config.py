# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Cyberdog2RoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_actions = 12
        num_observations = 45
        num_privileged_obs = 235 # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'trimesh'

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.28] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_abad_joint': 0.,   # [rad]
            'RL_abad_joint': 0.,   # [rad]
            'FR_abad_joint': -0.,  # [rad]
            'RR_abad_joint': -0.,  # [rad]

            'FL_hip_joint': -0.88,   # [rad]
            'RL_hip_joint': -0.88,   # [rad]
            'FR_hip_joint': -0.88,   # [rad]
            'RR_hip_joint': -0.88,   # [rad]

            'FL_knee_joint': 1.44,   # [rad]
            'RL_knee_joint': 1.44,   # [rad]
            'FR_knee_joint': 1.44,   # [rad]
            'RR_knee_joint': 1.44,   # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 17.}  # [N*m/rad]
        damping = {'joint': 0.3}  # [N*m*s/rad]
        action_scale = 0.2
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/cyberdog2/urdf/cyberdog2_with_head_angular.urdf"
        name = "cyberdog2"
        foot_name = "foot"
        penalize_contacts_on = ["hip", "knee", "base", "abad"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

    class domain_rand( LeggedRobotCfg.domain_rand):
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
  
    class rewards( LeggedRobotCfg.rewards ):
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0. #-0.2
            torques = -0.00001
            dof_vel = -0.
            dof_acc = -2.5e-7
            # dof_power = -2.0e-5
            base_height = -2.0 # -0. # 
            feet_air_time =  1.0
            # feet_clearance = -0.01
            collision = -1.
            feet_stumble = -0.0 
            action_rate = -0.01
            # action_smooth = -0.01
            stand_still = -0.
            power_distribution = -1.0e-6

        base_height_target = 0.26
        soft_dof_pos_limit = 0.9 # percentage of urdf limits, values above this limit are penalized
        max_contact_force = 100.
        only_positive_rewards = True
        # class scales( LeggedRobotCfg.rewards.scales ):
        #     pass

    class commands:
        curriculum = True
        max_curriculum = 2.0 # 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0] # [-1.5, 1.5] # min max [m/s]
            lin_vel_y = [-1.0, 1.0] # [-1.0, 1.0] # min max [m/s]
            ang_vel_yaw = [-2.0, 2.0] # [-2.0, 2.0] # min max [rad/s]
            heading = [-3.14, 3.14]

    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # terrain_proportions = [0.4, 0.3, 0.0, 0.0, 0.3]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

class Cyberdog2RoughCfgPPO( LeggedRobotCfgPPO ):
    class runner( LeggedRobotCfgPPO.runner ):
        max_iterations = 3000 # number of policy updates
        save_interval = 100 # check for potential saves every this many iterations
        run_name = ''
        experiment_name = 'rough_cyberdog2'
        load_run = -1
