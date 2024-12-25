# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs
from tqdm import tqdm, trange
import time
import imageio
import cv2
import subprocess

import metaworld_env
import drq_utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader, make_segment_replay_loader
from video import TrainVideoRecorder, VideoRecorder
from video_predictor import VideoPredictor
from drqv2 import DrQV2Agent

torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


def make_video_predictor(cfg):
    return VideoPredictor('cuda', cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        drq_utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                self.cfg.agent)

        self.video_predictor = make_video_predictor(self.cfg.world_model)

        self.timer = drq_utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create envs
        self.cfg.task_name = "-".join(self.cfg.task_name.split("_"))
        self.train_env = metaworld_env.make(self.cfg.task_name, self.cfg.frame_stack,
                                       self.cfg.action_repeat, self.cfg.seed, self.cfg.camera, self.cfg.duration, self.cfg.succ_bonus)
        self.eval_env = metaworld_env.make(self.cfg.task_name, self.cfg.frame_stack,
                                      self.cfg.action_repeat, self.cfg.seed, self.cfg.camera, self.cfg.duration, self.cfg.succ_bonus)
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')

        # for agent training
        demo_path = os.path.join(self.cfg.demo_path_prefix, self.cfg.task_name) if self.cfg.demo else None
        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            int(self.cfg.batch_size * self.cfg.real_ratio), self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount, demo_path)
        self._replay_iter = None

        self.imag_replay_storage = ReplayBufferStorage(data_specs,
                                                       self.work_dir / 'imag_buffer')

        self.imag_replay_loader = make_replay_loader(
            self.work_dir / 'imag_buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size - int(self.cfg.batch_size * self.cfg.real_ratio), self.cfg.replay_buffer_num_workers,
            False, self.cfg.nstep, self.cfg.discount)
        self._imag_replay_iter = None

        # for model training
        self.seg_replay_loader = make_segment_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.world_model.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount,
            self.cfg.gen_horizon + self.cfg.world_model.context_length, demo_path)
        self._seg_replay_iter = None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        while True:
            if self._replay_iter is None:
                self._replay_iter = iter(self.replay_loader)
            real_batch = next(self._replay_iter)
            if self.global_step * self.cfg.action_repeat >= self.cfg.start_mbpo:
                if self._imag_replay_iter is None:
                    self._imag_replay_iter = iter(self.imag_replay_loader)
                fake_batch = next(self._imag_replay_iter)
            else:
                fake_batch = next(self._replay_iter)
            mix_batch = [torch.cat([real, fake], 0) for real, fake in zip(real_batch, fake_batch)]
            yield mix_batch

    def eval(self):
        step, episode, total_reward, total_success = 0, 0, 0, 0
        eval_until_episode = drq_utils.Until(self.cfg.num_eval_episodes, bar_name='eval_eps')

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            episode_success = 0
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), drq_utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env, time_step.reward)
                total_reward += time_step.reward
                episode_success += time_step.success
                step += 1

            total_success += episode_success >= 1.0
            episode += 1
            # self.video_recorder.save(f'{self.global_frame}.mp4')
            self.video_recorder.save(f'{self.global_frame}.gif')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_success', total_success / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def generate(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        batch = next(self._replay_iter)
        policy = lambda obs, step: self.agent.act2(obs, self.global_step - 1, eval_mode=False)  # random action for init gen
        start = time.time()
        obss, actions, rewards = self.video_predictor.rollout(
            batch[0][:self.cfg.gen_batch], 
            policy,
            self.cfg.gen_horizon
        )
        for i in range(len(obss)):
            path = self.imag_replay_storage._store_episode(
            # path = self.imag_replay_loader.dataset._direct_store_episode(
                {
                    'action': actions[i].detach().cpu().numpy(),
                    'observation': (obss[i] * 255).detach().cpu().numpy().astype(np.uint8),
                    'reward': rewards[i].detach().cpu().numpy(),
                    'discount': np.ones_like(rewards[i].detach().cpu().numpy()),
                }
            )
            # save_gif
            if i % 10 == 0:
                gif_path = str(path).replace('imag_buffer', 'imag_gif').replace('.npz', '.gif')
                os.makedirs(os.path.dirname(gif_path), exist_ok=True)
                frames = []
                for j, obs in enumerate(obss[i]):
                    frame = (obs[-3:].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
                    frame = np.ascontiguousarray(frame)
                    cv2.putText(frame, f'{rewards[i][j].item():.2f}', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    frames.append(frame)
                imageio.mimsave(gif_path, frames, fps=4, loop=0)
        print(f'generate time: {time.time() - start}')
        return {
            "gen/reward_mean": rewards.mean().item(),
        }

    def validate(self, global_frame):
        if self._seg_replay_iter is None:
            self._seg_replay_iter = iter(self.seg_replay_loader)
        batch = next(self._seg_replay_iter)
        obs_gt = torch.cat([batch[0][:, :-2], batch[0][:, 1:-1], batch[0][:, 2:]], dim=2)
        action = batch[1][:, 2:]
        reward_gt = batch[2][:, 2:]
        policy = lambda obs, step: action[:, step].to(obs.device)
        start = time.time()
        obs_pred, _, reward_pred = self.video_predictor.rollout(
            obs_gt[:, 0], 
            policy,
            obs_gt.shape[1] - 1,
        )
        obs_mse = ((obs_pred[:, 1:] - (obs_gt[:, 1:]/255.).to(obs_pred.device)) ** 2).mean()
        reward_mse = ((reward_pred[:, 1:] - reward_gt[:, 1:].to(reward_pred.device)) ** 2).mean()
        print(f'validate time: {time.time() - start}')
        
        for i in range(obs_gt.shape[0]):
            gif_path = os.path.join(self.work_dir, 'validate_gif', f'val-sample-{global_frame}-{i}.gif')
            os.makedirs(os.path.dirname(gif_path), exist_ok=True)
            frames = []
            for t in range(obs_gt.shape[1]):
                frame = (obs_gt[i, t, -3:].permute(1,2,0).detach().cpu().numpy()).astype(np.uint8)
                frame = np.ascontiguousarray(frame)
                frame_pred = (obs_pred[i, t, -3:].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
                frame_pred = np.ascontiguousarray(frame_pred)
                frame_error = np.abs(frame.astype(float) - frame_pred.astype(float)).astype(np.uint8)
                if t > 0:
                    cv2.putText(frame, f'{reward_gt[i, t].item():.2f}', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    cv2.putText(frame_pred, f'{reward_pred[i, t].item():.2f}', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                
                frames.append(np.concatenate([frame, frame_pred, frame_error], axis=1))
            imageio.mimsave(gif_path, frames, fps=4, loop=0)
            
        return {
            "val/obs_mse": obs_mse.item(),
            "val/reward_mse": reward_mse.item(),
        }

    def train(self):
        assert self.cfg.num_seed_frames == self.cfg.agent.num_expl_steps * self.cfg.action_repeat

        # predicates
        train_until_step = drq_utils.Until(self.cfg.num_train_frames,
                                           self.cfg.action_repeat,
                                           bar_name='train_step')
        seed_until_step = drq_utils.Until(self.cfg.num_seed_frames,
                                          self.cfg.action_repeat)
        eval_every_step = drq_utils.Every(self.cfg.eval_every_frames,
                                          self.cfg.action_repeat)
        gen_every_step = drq_utils.Every(self.cfg.gen_every_steps,
                                         self.cfg.action_repeat)
        update_gen_every_step = drq_utils.Every(self.cfg.update_gen_every_step,
                                                self.cfg.action_repeat)

        episode_step, episode_reward, episode_success = 0, 0, 0
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)
        self.train_video_recorder.init(time_step.observation)
        metrics = None
        init_model = False
        init_gen = False
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                # self.train_video_recorder.save(f'{self.global_frame}.mp4')
                self.train_video_recorder.save(f'{self.global_frame}.gif')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_success', episode_success >= 1.0)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)

                # reset env
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                if self.cfg.save_snapshot and self._global_episode % 10 == 0:
                    self.save_snapshot()
                    self.video_predictor.save_snapshot(self.work_dir)
                episode_step = 0
                episode_reward = 0
                episode_success = 0
                
                if not seed_until_step(self.global_step) and self._global_episode % 5 == 0:
                    metrics = self.validate(self.global_frame)
                    self.logger.log_metrics(metrics, self.global_frame, ty='eval')

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            # sample action
            with torch.no_grad(), drq_utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                if not init_model:
                    # init train the model
                    for _ in trange(self.cfg.init_update_gen_steps):
                        if self._seg_replay_iter is None:
                            self._seg_replay_iter = iter(self.seg_replay_loader)
                        batch = next(self._seg_replay_iter)
                        metrics = self.video_predictor.train(batch)
                        if _ % 10 == 0:
                            metrics = {k + "_init": v for k, v in metrics.items()}
                            self.logger.log_metrics(metrics, _, ty='train')
                    self.video_predictor.save_snapshot(self.work_dir, suffix='_init')
                    
                    metrics = self.validate(self.global_frame)
                    self.logger.log_metrics(metrics, self.global_frame, ty='eval')

                    init_model = True
                else:
                    # update the model
                    if update_gen_every_step(self.global_step):
                        for _ in range(self.cfg.update_gen_times):
                            if self._seg_replay_iter is None:
                                self._seg_replay_iter = iter(self.seg_replay_loader)
                            batch = next(self._seg_replay_iter)
                            metrics = self.video_predictor.train(batch, update_tokenizer=self.global_step % (self.cfg.update_tokenizer_every_step // self.cfg.action_repeat) == 0)
                        self.logger.log_metrics(metrics, self.global_frame, ty='train')
                
                if self.global_step * self.cfg.action_repeat >= self.cfg.start_mbpo and not init_gen:
                    for _ in trange(self.cfg.init_gen_times):
                        self.generate()
                    init_gen = True

                for _ in range(self.cfg.agent_update_times):
                    metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')
                
                # generate fake transitions
                if self.global_step * self.cfg.action_repeat >= self.cfg.start_mbpo and gen_every_step(self.global_step):
                    metrics = self.generate()
                    self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            episode_success += time_step.success
            if time_step.last():
                self.last_episode = self.replay_storage.add(time_step)
            else:
                self.replay_storage.add(time_step)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(config_path='cfgs', config_name='mbpo_config')
def main(cfg):
    root_dir = Path.cwd()
    workspace = Workspace(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()

    # snapshot src code
    os.makedirs('src', exist_ok=True)
    os.system(f"rsync -rv --exclude-from=../../../.gitignore ../../.. src")

    workspace.train()


if __name__ == '__main__':
    main()
