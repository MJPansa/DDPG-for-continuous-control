from unityagents import UnityEnvironment
from collections import deque
import torch as T
import numpy as np


class EnvWrapper:
    """A wrapper for the unity environment which implements functionalies similar to openai gym

    Params
    ======
        path(string): relative/absolute path to env executable
    """
    def __init__(self, path, no_graphics=True):
        self.env = UnityEnvironment(file_name=path, no_graphics=no_graphics)
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        self.env_info = None
        self.reset()

        self.action_space = self.brain.vector_action_space_size
        self.observation_space = self.env_info.vector_observations.shape[1]

    def step(self, actions):
        self.env_info = self.env.step(actions)[self.brain_name]
        next_state = self.env_info.vector_observations
        reward = self.env_info.rewards
        done = self.env_info.local_done
        return next_state, reward, done, None

    def reset(self):
        self.env_info = self.env.reset(train_mode=True)[self.brain_name]
        return self.env_info.vector_observations


class DDPGExperienceBuffer:
    """A replay buffer which stores the experiences (s,a,r,d,s') and allows for sampling

    Params
    ======
        size (int): maximum size of buffer
        bs (int): batchsize for sampling from buffer
        threshold (float): percentage of buffersize that has to be reached before sampling
        device (string): 'cuda' or 'cpu'
    """
    def __init__(self, size, bs, threshold, device):
        self.size = size
        self.bs = bs
        self.threshold_v = threshold
        self.device = device

        self.states = deque(maxlen=self.size)
        self.actions = deque(maxlen=self.size)
        self.rewards = deque(maxlen=self.size)
        self.dones = deque(maxlen=self.size)
        self.next_states = deque(maxlen=self.size)

    def add(self, state, action, reward, done, next_state):
        for n in range(20):
            self.states.append(state[n, :])
            self.actions.append(action[n, :])
            self.rewards.append(reward[n, :])
            self.dones.append(done[n, :])
            self.next_states.append(next_state[n, :])

    def draw(self):
        random = np.random.permutation(np.arange(len(self)))[:self.bs]
        return [T.stack([x[i] for i in random], dim=0).to(self.device) for x in [self.states, self.actions, self.rewards,
                                                                    self.dones, self.next_states]]
    @property
    def threshold(self):
        return True if self.threshold_v < self.__len__() / self.size else False

    def __len__(self):
        return len(self.states)
