import torch.nn as nn
import torch
import structures.const as const
import random
import _thread

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
#        n0, F1, S1, P1 = cell_number, 3, 1, 0
#        self.conv1 = nn.Conv2d(2, 16, kernel_size=F1, stride=S1, padding=P1)
#        self.bn1 = nn.BatchNorm2d(16)
#        n1, F2, S2, P2 = int((n0 - F1 + 2 * P1) / S1 + 1), 3, 1, 0
#        self.conv2 = nn.Conv2d(16, 32, kernel_size=F2, stride=S2, padding=P2)
#        self.bn2 = nn.BatchNorm2d(32)
#        n2 = int((n1 - F2 + 2 * P2)/S1 + 1)
#        n3 = n2 * n2 * 32
#        # self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
#        # self.bn3 = nn.BatchNorm2d(32)
#        self.l1 = nn.Linear(n3, 16)
#        self.head = nn.Linear(16, 4)
        self.LOCK = _thread.allocate_lock()
        self.model16 = nn.Sequential(
            nn.Conv2d(1, 32, 16),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.model8 = nn.Sequential(
            nn.Conv2d(1, 32, 8, stride = 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.model4 = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride = 2),
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
            nn.Linear(64, 4),
            nn.Softmax())


    def forward(self, x):
        f16 = self.model16(x).squeeze(3).squeeze(2)
        f8 = self.model8(x).squeeze(3).squeeze(2) 
        f4 = self.model4(x).squeeze(3).squeeze(2) 
        x = torch.cat((f16, f8, f4), dim = 1)
        return self.linear(x)


    def eps_greedy(self, state, iter):
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
        with self.LOCK:
            self.eval()
            q_value = self(state)
            self.train()
        return q_value.max(1)[1].data[0]
        
