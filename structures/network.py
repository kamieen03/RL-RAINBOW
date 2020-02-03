import torch.nn as nn
import torch
import structures.const as const
import random
import _thread
from collections import deque

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
#        self.model16 = nn.Sequential(
#            nn.Conv2d(4, 32, 16),
#            nn.BatchNorm2d(32),
#            nn.ReLU())
#        self.model8 = nn.Sequential(
#            nn.Conv2d(4, 32, 8, stride = 2),
#            nn.BatchNorm2d(32),
#            nn.ReLU(),
#            nn.Conv2d(32, 64, 3),
#            nn.BatchNorm2d(64),
#            nn.ReLU(),
#            nn.Conv2d(64, 64, 3),
#            nn.BatchNorm2d(64),
#            nn.ReLU())
        self.model4 = nn.Sequential(
            nn.Conv2d(4, 32, 4, stride = 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            nn.ReLU())
        self.V_f = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1))
        self.A_f = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4))


    def forward(self, x):
        #f16 = self.model16(x).squeeze(3).squeeze(2)
        #f8 = self.model8(x).squeeze(3).squeeze(2) 
        f4 = self.model4(x).squeeze(3).squeeze(2) 
        #x = torch.cat((f16, f8, f4), dim = 1)
        V_vals = self.V_f(f4)
        A_vals = self.A_f(f4)
        return V_vals + A_vals - A_vals.mean(dim=1).unsqueeze(1)


    def eps_greedy(self, state, iter):
        if type(state) == deque:
            state = torch.cat(tuple(state), dim = 1)
        if random.random() > const.epsilon(iter):
            q_value = self(state)
            action = q_value.max(1)[1].data[0]
        else:
            action = torch.tensor(random.randrange(4))
        return action

    def greedy(self, state):
        if type(state) == deque:
            state = torch.cat(tuple(state), dim = 1)
        self.eval()
        with torch.no_grad():
            q_value = self(state)
        if random.random() < 0.03:
            return q_value[0].sort()[1][-2]
        return q_value.max(1)[1].data[0]
        

