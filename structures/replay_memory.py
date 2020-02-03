from collections import deque
import random
import torch

from const import BATCH_SIZE

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state = torch.cat(tuple(state), dim=1)
        next_state = torch.cat(tuple(next_state), dim=1)
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self):
        state, action, reward, next_state, done = zip(*random.sample(self.memory, BATCH_SIZE))
        return torch.cat(state).cuda(), torch.tensor(action).cuda(), \
               torch.tensor(reward).cuda(), torch.cat(next_state).cuda(),\
               torch.tensor(done).cuda()
    
    def __len__(self):
        return len(self.memory)

