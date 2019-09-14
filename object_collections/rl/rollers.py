"""
Ways of gathering state transitions to store in a replay
buffer.

Stolen from: http://github.com/unixpickle/anyrl-py
"""
import numpy as np

from abc import ABC, abstractmethod
from collections import OrderedDict
import time

from anyrl.rollouts.util import reduce_states, inject_state, reduce_model_outs
from anyrl.rollouts import empty_rollout
import itertools
from object_collections.rl.mpi_util import RunningMeanVar

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """
    def __init__(self, obs_dim, act_dim, FLAGS):
        self.FLAGS = FLAGS

        self.bufs = {}
        dyn_size = lambda: np.zeros([self.FLAGS['replay_size'], self.FLAGS['embed_shape']], dtype=np.float32)

        if self.FLAGS['use_embed']:
            for key in self.FLAGS['embeds']:
                self.bufs[key] = dyn_size()
        else:
            self.bufs['s'] = np.zeros([self.FLAGS['replay_size'], obs_dim], dtype=np.float32)
            self.bufs['s_next'] = np.zeros([self.FLAGS['replay_size'], obs_dim], dtype=np.float32)

        self.bufs['a'] = np.zeros([self.FLAGS['replay_size'], act_dim], dtype=np.float32)
        self.bufs['r'] = np.zeros(self.FLAGS['replay_size'], dtype=np.float32)
        self.bufs['d'] = np.zeros(self.FLAGS['replay_size'], dtype=np.float32)

        if not self.FLAGS['abs_noise']:
            self.running_mean_vars = {key: RunningMeanVar() for key in self.bufs}
        self.ptr, self.size, self.max_size = 0, 0, self.FLAGS['replay_size']

    def store(self, sarsd):
        raise Exception('not supported anymore')
        for key in self.bufs:
            self.bufs[key][self.ptr] = sarsd[key]

        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def store_batch(self, sarsd_batch):
        """
        With batch, we sometimes have to write partially to the end and partially to the beginning.

        This handles that wrap-around logic 
        Confidence 4: I am confident, but not absolutely certain this is correct
        """
        batch_len = len(sarsd_batch['d'])
        end_idx = self.ptr + batch_len

        if not self.FLAGS['abs_noise']:
            for key in sarsd_batch:
                self.running_mean_vars[key].update_from_moments(np.mean(sarsd_batch[key], axis=0), np.var(sarsd_batch[key], axis=0), batch_len)

        if end_idx <= self.max_size:
            for key in self.bufs: self.bufs[key][self.ptr:end_idx] = sarsd_batch[key]
            self.ptr = (self.ptr + batch_len) % self.max_size
        else:
            overflow = (end_idx - self.max_size)
            top_off = batch_len - overflow

            for key in self.bufs: self.bufs[key][self.ptr:self.ptr+top_off] = sarsd_batch[key][:top_off]
            for key in self.bufs: self.bufs[key][:overflow] = sarsd_batch[key][top_off:]

            self.ptr = overflow
        self.size = min(self.size+batch_len, self.max_size)

    def sample_batch(self, batch_size=32, noise=0.0):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = {key: self.bufs[key][idxs] for key in self.bufs}
        if noise != 0.0:
            for key in batch:
                if key is not 'r' and key is not 'd':
                    if self.FLAGS['abs_noise']:
                        batch[key] = batch[key] + np.random.normal(scale=noise, size=batch[key].shape)
                    else:
                        batch[key] = batch[key] + noise * self.running_mean_vars[key].var * np.random.randn(*batch[key].shape)
        return batch



"""
Ways of gathering rollouts.
"""


class Roller(ABC):
    """
    An object that gathers rollouts by running a model.
    """
    @abstractmethod
    def rollouts(self, mode='explore', model=None, scripted=None, p=None):
        """
        Return a list of Rollout objects.
        """
        pass


class BasicRoller(Roller):
    """
    Gathers episode rollouts from a Gym environment.
    """

    def __init__(self, env, model, min_episodes=1, min_steps=1):
        self.env = env
        self.model = model
        self.min_episodes = min_episodes
        self.min_steps = min_steps

    def rollouts(self, mode='explore', model=None, scripted=None, p=None):
        """
        Gather episodes until both self.min_episodes and
        self.min_steps are satisfied.
        """
        model = model or self.model
        episodes = []
        num_steps = 0
        while num_steps < self.min_steps and len(episodes) < self.min_episodes:
            states = self.model.start_state(1)
            rollout = empty_rollout(states)
            obs = self.env.reset()
            while True:
                if rollout.num_steps > self.min_steps:
                    break
                rollout.observations.append(obs)

                if p is not None and np.random.uniform(0,1) < p:
                    model_out = scripted.step([obs], states, mode=mode)
                else:
                    model_out = model.step([obs], states, mode=mode)

                rollout.model_outs.append(model_out)
                states = model_out['states']
                obs, rew, done, info = self.env.step(model_out['actions'][0])
                rollout.rewards.append(rew)
                rollout.infos.append(info)
                if done:
                    break
            num_steps += rollout.num_steps
            rollout.end_time = time.time()
            episodes.append(rollout)
        return episodes


class TruncatedRoller(Roller):
    """
    Gathers a fixed number of timesteps from each
    environment in a BatchedEnv.
    """

    def __init__(self, batched_env, model, num_timesteps, drop_states=False):
        """
        Create a new TruncatedRoller.

        Args:
          batched_env: a BatchedEnv to interact with.
          model: a Model to use for interaction.
          num_timesteps: the number of timesteps to run
            each sub-environment for.
          drop_states: if True, set model_outs['states']
            to None in the rollouts to save memory.
        """
        self.batched_env = batched_env
        self.model = model
        self.num_timesteps = num_timesteps
        self.drop_states = drop_states

        # These end up being batches of sub-batches.
        # Each sub-batch corresponds to a sub-batch of
        # environments.
        self._last_states = None
        self._last_obs = None
        self._prev_steps = None
        self._prev_reward = None

    def reset(self):
        """
        Reset the environments, model states, and partial
        trajectory information.

        This needn't be called on new TruncatedRollers.
        """
        inner_dim = self.batched_env.num_envs_per_sub_batch
        outer_dim = self.batched_env.num_sub_batches
        self._last_obs = []
        self._prev_steps = [[0] * inner_dim for _ in range(outer_dim)]
        self._prev_reward = [[0] * inner_dim for _ in range(outer_dim)]
        for i in range(outer_dim):
            self.batched_env.reset_start(sub_batch=i)
        for i in range(outer_dim):
            self._last_obs.append(self.batched_env.reset_wait(sub_batch=i))
        self._last_states = [self.model.start_state(inner_dim)
                             for _ in range(outer_dim)]

    def rollouts(self, mode='explore', model=None, scripted=None, p=None):
        """
        Gather (possibly truncated) rollouts.
        """
        model = model or self.model
        if self._last_states is None:
            self.reset()
        completed_rollouts = []
        running_rollouts = self._starting_rollouts()
        for _ in range(self.num_timesteps):
            self._step(completed_rollouts, running_rollouts, mode=mode, model=model)
        self._step(completed_rollouts, running_rollouts, final_step=True, mode=mode, model=model)
        self._add_truncated(completed_rollouts, running_rollouts)
        return completed_rollouts

    def _starting_rollouts(self):
        """
        Create empty rollouts with the start states and
        initial observations.
        """
        rollouts = []
        for batch_idx, states in enumerate(self._last_states):
            rollout_batch = []
            for env_idx in range(self.batched_env.num_envs_per_sub_batch):
                sub_state = reduce_states(states, env_idx)
                prev_steps = self._prev_steps[batch_idx][env_idx]
                prev_reward = self._prev_reward[batch_idx][env_idx]
                rollout = empty_rollout(sub_state,
                                        prev_steps=prev_steps,
                                        prev_reward=prev_reward)
                rollout_batch.append(rollout)
            rollouts.append(rollout_batch)
        return rollouts

    def _step(self, completed, running, final_step=False, mode='explore', model=None, scripted=None, p=None):
        """
        Wait for the previous batched step to complete (or
        use self._last_obs) and start a new step.

        Updates the running rollouts to reflect new steps
        and episodes.

        Returns the newest batch of model outputs.
        """
        model = model or self.model
        for batch_idx, obses in enumerate(self._last_obs):
            if obses is None:
                step_out = self.batched_env.step_wait(sub_batch=batch_idx)
                obses, rews, dones, infos = step_out
                for env_idx, (rew, done, info) in enumerate(zip(rews, dones, infos)):
                    running[batch_idx][env_idx].rewards.append(rew)
                    running[batch_idx][env_idx].infos.append(info)
                    if done:
                        self._complete_rollout(completed, running, batch_idx, env_idx)
                    else:
                        self._prev_steps[batch_idx][env_idx] += 1
                        self._prev_reward[batch_idx][env_idx] += rew

            states = self._last_states[batch_idx]

            if p is not None and np.random.uniform(0,1) < p:
                model_outs = scripted.step(obses, states, mode=mode)
            else:
                model_outs = model.step(obses, states, mode=mode)

            for env_idx, (obs, rollout) in enumerate(zip(obses, running[batch_idx])):
                reduced_out = self._reduce_model_outs(model_outs, env_idx)
                rollout.observations.append(obs)
                rollout.model_outs.append(reduced_out)

            if final_step:
                self._last_obs[batch_idx] = obses
            else:
                self._last_states[batch_idx] = model_outs['states']
                self._last_obs[batch_idx] = None
                self.batched_env.step_start(model_outs['actions'], sub_batch=batch_idx)

    def _complete_rollout(self, completed, running, batch_idx, env_idx):
        """
        Finalize a rollout and start a new rollout.
        """
        running[batch_idx][env_idx].end_time = time.time()
        completed.append(running[batch_idx][env_idx])
        for prev in [self._prev_steps, self._prev_reward]:
            prev[batch_idx][env_idx] = 0
        start_state = self.model.start_state(1)
        inject_state(self._last_states[batch_idx], start_state, env_idx)

        rollout = empty_rollout(start_state)
        running[batch_idx][env_idx] = rollout

    def _add_truncated(self, completed, running):
        """
        Add partial but non-empty rollouts to completed.
        """
        for sub_running in running:
            for rollout in sub_running:
                if rollout.num_steps > 0:
                    rollout.end_time = time.time()
                    completed.append(rollout)

    def _reduce_model_outs(self, model_outs, env_idx):
        """
        Reduce the model_outs to be put into a Rollout.
        """
        res = reduce_model_outs(model_outs, env_idx)
        if self.drop_states:
            res['states'] = None
        return res


class EpisodeRoller(TruncatedRoller):
    """
    Gather rollouts from a BatchedEnv with step and
    episode quotas.

    An EpisodeRoller does not have any bias towards
    shorter episodes.
    As a result, it must gather at least as many episodes
    as there are environments in batched_env.
    """

    def __init__(self, batched_env, model, min_episodes=1, min_steps=1, drop_states=False):
        """
        Create a new EpisodeRoller.

        Args:
          batched_env: a BatchedEnv to interact with.
          model: a Model to use for interaction.
          min_episodes: the minimum number of episodes to
            rollout.
          min_steps: the minimum number of timesteps to
            rollout, across all environments.
          drop_states: if True, set model_outs['states']
            to None in the rollouts to save memory.
        """
        self.min_episodes = min_episodes
        self.min_steps = min_steps

        # A batch of booleans in the shape of the envs.
        # An environment gets masked out once we have met
        # all the criteria and aren't looking for another
        # episode.
        self._env_mask = None

        super().__init__(batched_env, model, 0, drop_states=drop_states)

    def reset(self):
        super().reset()
        inner_dim = self.batched_env.num_envs_per_sub_batch
        outer_dim = self.batched_env.num_sub_batches
        self._env_mask = [[True] * inner_dim for _ in range(outer_dim)]

    def rollouts(self, mode='explore', model=None, scripted=None, p=None):
        """
        Gather full-episode rollouts.
        """
        model = model or self.model
        self.reset()
        completed_rollouts = []
        running_rollouts = self._starting_rollouts()
        while self._any_envs_running():
            self._step(completed_rollouts, running_rollouts, mode=mode, model=model, scripted=scripted, p=p)
        # Make sure we are ready for the next reset().
        for batch_idx in range(self.batched_env.num_sub_batches):
            self.batched_env.step_wait(sub_batch=batch_idx)
        return completed_rollouts

    def _complete_rollout(self, completed, running, batch_idx, env_idx):
        comp_dest = completed
        if not self._env_mask[batch_idx][env_idx]:
            comp_dest = []
        super()._complete_rollout(comp_dest, running, batch_idx, env_idx)
        if self._criteria_met(completed):
            self._env_mask[batch_idx][env_idx] = False

    def _criteria_met(self, completed):
        """
        Check if the stopping criteria are met.
        """
        total_steps = sum([r.num_steps for r in completed])
        total_eps = len(completed)
        return total_steps >= self.min_steps and total_eps >= self.min_episodes

    def _any_envs_running(self):
        """
        Check if any environment is not masked out.
        """
        return any([any(masks) for masks in self._env_mask])