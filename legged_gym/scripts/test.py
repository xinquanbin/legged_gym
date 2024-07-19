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

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

import time
import lcm

from legged_gym.lcm_type.rl_vae_action_lcmt import rl_vae_action_lcmt
from legged_gym.lcm_type.rl_vae_obs_lcmt import rl_vae_obs_lcmt

SUB_CHANNEL = 'rl_vae_obs'
PUB_CHANNEL = 'rl_vae_act'
LCM_URL = 'udpm://239.255.76.67:7688?ttl=255'

class TorchModel:
    def __init__(self, args):
        args.headless = True
        
        self.env_cfg, self.train_cfg = task_registry.get_cfgs(name=args.task)

        # override some parameters for testing
        self.env_cfg.env.num_envs = 1 # min(env_cfg.env.num_envs, 50)
        self.env_cfg.terrain.num_rows = 5
        self.env_cfg.terrain.num_cols = 5
        self.env_cfg.terrain.curriculum = False
        self.env_cfg.noise.add_noise = False
        self.env_cfg.domain_rand.randomize_friction = False
        self.env_cfg.domain_rand.push_robots = False

        # prepare environment
        self.env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=self.env_cfg)
        self.obs = self.env.get_observations()

        # load policy
        self.train_cfg.runner.resume = True
        self.ppo_runner, self.train_cfg = task_registry.make_alg_runner(env=self.env, name=args.task, args=args, train_cfg=self.train_cfg)

        self.ppo_runner.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        self.ppo_runner.alg.actor_critic.to(self.env.device)
    
    def predict(self, obs_data):
        self.obs = torch.tensor(obs_data, dtype=torch.float).reshape(1, -1).to(self.env.device)
        actions_mean = self.ppo_runner.alg.actor_critic.actor(self.obs.detach())
        return actions_mean.detach().cpu().numpy().reshape(-1)


class InferenceServer:
    def __init__(self, model, sub_channel, pub_channel, lcm_url):
        self.model = model
        self.sub_channel = sub_channel
        self.pub_channel = pub_channel
        self.lc = lcm.LCM(lcm_url)

    def model_inference(self, channel, data):
        # start_time = time.time()
        obs = rl_vae_obs_lcmt.decode(data).obs
        action = self.model.predict(obs)
        
        action_msg = rl_vae_action_lcmt()
        action_msg.action = list(action)
        self.lc.publish(self.pub_channel, action_msg.encode())
        # end_time = time.time()
        # inference_time = (end_time - start_time) * 1000
        # print(f'Average inference time: {inference_time} ms')

    def run(self):
        sub = self.lc.subscribe(self.sub_channel, self.model_inference)

        try:
            print('Inference server is running...')
            while True:
                self.lc.handle()
        except KeyboardInterrupt:
            pass



def test_inference_time(model: TorchModel, count):
    start_time = time.time()
    OBS_DIM = 45
    for _ in range(count):
        obs = np.zeros((1, OBS_DIM), dtype=np.float32)
        _ = model.predict(obs)
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000 / count
    print(f'Average inference time for {count} runs: {inference_time} ms')

def main(args):
    model = TorchModel(args)
    
    if args.test:
        test_inference_time(model, 100)
        return

    inference = InferenceServer(model, SUB_CHANNEL, PUB_CHANNEL, LCM_URL)
    try:
        inference.run()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    args = get_args()
    main(args)