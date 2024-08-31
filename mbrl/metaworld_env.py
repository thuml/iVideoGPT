# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from collections import deque
from typing import Any, NamedTuple

import dm_env
import gym
import numpy as np
from dm_env import StepType, specs


_ACTION_SPEC_MUST_BE_BOUNDED_ARRAY = (
    "`env.action_spec()` must return a single `BoundedArray`, got: {}.")
_MUST_BE_FINITE = "All values in `{name}` must be finite, got: {bounds}."
_MUST_BROADCAST = (
    "`{name}` must be broadcastable to shape {shape}, got: {bounds}.")


class ActionScaleWrapper(dm_env.Environment):
    """Wraps a control environment to rescale actions to a specific range."""
    __slots__ = ("_action_spec", "_env", "_transform")

    def __init__(self, env, minimum, maximum):
        """Initializes a new action scale Wrapper.

        Args:
          env: Instance of `dm_env.Environment` to wrap. Its `action_spec` must
            consist of a single `BoundedArray` with all-finite bounds.
          minimum: Scalar or array-like specifying element-wise lower bounds
            (inclusive) for the `action_spec` of the wrapped environment. Must be
            finite and broadcastable to the shape of the `action_spec`.
          maximum: Scalar or array-like specifying element-wise upper bounds
            (inclusive) for the `action_spec` of the wrapped environment. Must be
            finite and broadcastable to the shape of the `action_spec`.

        Raises:
          ValueError: If `env.action_spec()` is not a single `BoundedArray`.
          ValueError: If `env.action_spec()` has non-finite bounds.
          ValueError: If `minimum` or `maximum` contain non-finite values.
          ValueError: If `minimum` or `maximum` are not broadcastable to
            `env.action_spec().shape`.
        """
        action_spec = env.action_spec()
        if not isinstance(action_spec, specs.BoundedArray):
            raise ValueError(_ACTION_SPEC_MUST_BE_BOUNDED_ARRAY.format(action_spec))

        minimum = np.array(minimum)
        maximum = np.array(maximum)
        shape = action_spec.shape
        orig_minimum = action_spec.minimum
        orig_maximum = action_spec.maximum
        orig_dtype = action_spec.dtype

        def validate(bounds, name):
            if not np.all(np.isfinite(bounds)):
                raise ValueError(_MUST_BE_FINITE.format(name=name, bounds=bounds))
            try:
                np.broadcast_to(bounds, shape)
            except ValueError:
                raise ValueError(_MUST_BROADCAST.format(
                    name=name, bounds=bounds, shape=shape))

        validate(minimum, "minimum")
        validate(maximum, "maximum")
        validate(orig_minimum, "env.action_spec().minimum")
        validate(orig_maximum, "env.action_spec().maximum")

        scale = (orig_maximum - orig_minimum) / (maximum - minimum)

        def transform(action):
            new_action = orig_minimum + scale * (action - minimum)
            return new_action.astype(orig_dtype, copy=False)

        dtype = np.result_type(minimum, maximum, orig_dtype)
        self._action_spec = action_spec.replace(
            minimum=minimum, maximum=maximum, dtype=dtype)
        self._env = env
        self._transform = transform

    def step(self, action):
        return self._env.step(self._transform(action))

    def reset(self):
        return self._env.reset()

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any
    success: Any
    state: Any = None

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key='pixels'):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()

        pixels_shape = wrapped_obs_spec.shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = specs.BoundedArray(shape=np.concatenate(
            [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name='observation')

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0,
                                success=time_step.success or 0.0,
                                state=time_step.state)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


def make(name, frame_stack, action_repeat, seed, camera, duration, succ_bonus):
    env = MetaWorld(name, action_repeat=action_repeat, seed=seed, camera=camera, duration=duration, succ_bonus=succ_bonus)
    pixels_key = 'observation'
    # add wrappers
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionScaleWrapper(env, minimum=-1.0, maximum=+1.0)
    # stack several frames
    env = FrameStackWrapper(env, frame_stack, pixels_key)
    env = ExtendedTimeStepWrapper(env)
    return env


class MetaWorldTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    success: Any
    state: Any = None

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)


class MetaWorld(dm_env.Environment):

    def __init__(self, name, seed=None, action_repeat=1, size=(64, 64), camera=None, use_gripper=False, duration=500, succ_bonus=0.0):
        import metaworld
        from metaworld.envs import (
            ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
        )

        os.environ["MUJOCO_GL"] = "egl"

        task = f"{name}-v2-goal-observable"
        env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task]
        self._env = env_cls(seed=seed)
        self._env._freeze_rand_vec = False
        
        import mujoco
        self._env.render_mode = "rgb_array"
        self._env.mujoco_renderer.camera_id = mujoco.mj_name2id(
            self._env.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            "corner",
        )
        self._env.mujoco_renderer.height = size[0]
        self._env.mujoco_renderer.width = size[1]
        
        self._size = size
        self._action_repeat = action_repeat
        self._use_gripper = use_gripper
        self._steps = None
        self._duration = duration
        self._succ_bonus = succ_bonus

        self._camera = camera

    def observation_spec(self):
        return specs.BoundedArray(shape=self._size + (3,), dtype=np.uint8, minimum=0, maximum=255, name='observation')

    def action_spec(self):
        return specs.BoundedArray(shape=self._env.action_space.shape,
                                  dtype=np.float32,
                                  minimum=self._env.action_space.low.min(),
                                  maximum=self._env.action_space.high.max(),
                                  name='action')

    def step(self, action):
        assert self._steps is not None, "Must reset environment."
        assert np.isfinite(action).all(), action
        reward = 0.0
        success = 0.0
        for _ in range(self._action_repeat):
            state, rew, done, truncate, info = self._env.step(action)
            success += float(info["success"])
            reward += rew or 0.0
        success = float(success >= 1.0)
        if success == 1.0:
            reward += self._succ_bonus
        assert success in [0.0, 1.0]
        # image is upside-down https://github.com/shikharbahl/neural-dynamic-policies/blob/main/metaworld/metaworld/core/gym_to_multi_env.py#L79
        image = self._env.render()[::-1]  
        self._steps += 1
        if self._steps >= self._duration:
            done = True
            self._steps = None
        return MetaWorldTimeStep(step_type=dm_env.StepType.LAST if done else dm_env.StepType.MID, reward=reward, discount=1,observation=image, success=success, state=state)

    def reset(self):
        self._steps = 0
        if self._camera == "corner2":
            self._env.model.cam_pos[2][:] = [0.75, 0.075, 0.7]
        self._env.reset()
        state, rew, done, truncate, info = self._env.step(np.zeros(self._env.action_space.shape))
        image = self._env.render()[::-1]
        return MetaWorldTimeStep(step_type=dm_env.StepType.FIRST, reward=0, discount=1, observation=image, success=0.0,
                                 state=state)

    def render(self, mode='offscreen'):
        return self._env.render()[::-1]

    def __getattr__(self, name):
        return getattr(self._env, name)
