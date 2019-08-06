import torch.nn as nn
import torch
import structures.const as const
import random
import _thread
from collections import deque

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.LOCK = _thread.allocate_lock()
        self.model16 = nn.Sequential(
            nn.Conv2d(4, 32, 16),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.model8 = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride = 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.model4 = nn.Sequential(
            nn.Conv2d(4, 32, 4, stride = 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.linear = nn.Sequential(
            nn.Linear(160, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 4))


    def forward(self, x):
        f16 = self.model16(x).squeeze(3).squeeze(2)
        f8 = self.model8(x).squeeze(3).squeeze(2) 
        f4 = self.model4(x).squeeze(3).squeeze(2) 
        x = torch.cat((f16, f8, f4), dim = 1)
        return self.linear(x)


    def eps_greedy(self, state, iter):
        if type(state) == deque:
            state = torch.cat(tuple(state), dim = 1)
        with self.LOCK:
            if random.random() > const.epsilon(iter):
                self.eval()
                q_value = self(state)
                self.train()
                action = q_value.max(1)[1].data[0]
            else:
                action = torch.tensor(random.randrange(4))
        return action

    def greedy(self, state):
        if type(state) == deque:
            state = torch.cat(tuple(state), dim = 1)
        with self.LOCK:
            self.eval()
            q_value = self(state)
            self.train()
        return q_value.max(1)[1].data[0]
        
