from __future__ import annotations
from collections import deque
import random
import numpy as np


class Buffer:
    def __init__(self, max_size, sample_size=128):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.sample_size = sample_size
    
    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self):
        # TODO: make this faster; less correlated sampling (see batch norm 2015 paper)
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, self.sample_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)

    @property
    def batch_size(self):
        return self.sample_size