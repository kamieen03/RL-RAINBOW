import random
import tkinter
import torch
import torch.optim as optim
from collections import deque

import structures.const as const
from structures.network import DQN
from structures.sumtree import SumTree

from snakegame.game import Game
from snakegame.painter import Painter

class Agent:
    def __init__(self):
        self.policy_net = DQN()
        self.target_net = DQN()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.cuda()
        self.target_net.cuda()
        self.memory = SumTree(16384)
        self.optimizer = optim.Adam(self.policy_net.parameters())

        self.env = Game(const.BOARD_SIZE)
        self.state = deque([self.env.get_state() for _ in range(4)], maxlen = 4)
        self.iter = 0

    def test(self):
        for i in range(10):
            game = Game(const.BOARD_SIZE)
            game.newgame()
            state = deque(maxlen = 4)
            state.extend([game.get_state() for _ in range(4)])
            ep_length, reward = 0, 0
            while True:
                state.append(game.get_state())
                action = self.target_net.greedy(state)
                _dict = game.step(action)
                reward += _dict['reward']
                ep_length += 1
                if _dict['dead']:
                    print(game.snake.eaten)
                    break

    def train(self):
        self.policy_net.train()
        self.target_net.eval()
        while True:
            action = self.policy_net.eps_greedy(self.state, self.iter) 
            _dict = self.env.step(action)
            reward = torch.tensor(_dict['reward'], dtype = torch.float)
            dead = torch.tensor(_dict['dead'], dtype = torch.float)
            next_state = self.state.copy()
            next_state.append(self.env.get_state())
            self.memory.push(self.state, action, reward, next_state, dead)
            self.state = next_state
            
            if len(self.memory) >= const.BATCH_SIZE: 
                self._perform_gradient_step()
            self.iter += 1

            if dead:
                self.env.newgame()
                self.state.clear()
                self.state.extend([self.env.get_state() for _ in range(4)])
            if self.iter % 100 == 0:
                print(self.iter)
                self.target_net.load_state_dict(self.policy_net.state_dict())
                torch.save(self.target_net.state_dict(), "model.pth")
                self.target_net.eval()
                self.test()



    def _perform_gradient_step(self):
        states, actions, rewards, next_states, deaths, indices, weights = self.memory.sample()

        self.optimizer.zero_grad()
        max_actions = self.policy_net(next_states).max(1)[1]
        with torch.no_grad():
            next_q_vals = self.target_net(next_states).gather(1, max_actions.unsqueeze(1)).squeeze(1)

        q_vals = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        target_vals = rewards + const.GAMMA * next_q_vals * (1 - deaths)
        
        loss  = (target_vals - q_vals).pow(2) * torch.tensor(weights, dtype = torch.float).cuda()
        prios = loss + 10**(-5)
        loss  = loss.mean()

        loss.backward()
        self.optimizer.step()

        self.memory.update_priorities(indices, prios.data.cpu().numpy())


    def visible_play(self):
        def start():
            game.newgame()
            state.clear()
            state.extend([game.get_state() for _ in range(4)])
            painter.drawnewgame(game.snake)
            frame.after(0, step)
        def step():
            state.append(game.get_state())
            action = self.target_net.greedy(state)
            _dict = game.step(action)
            if _dict['dead']:
                frame.after(3000, start)
                return
            painter.draw_changes(game.grid.get_reward())
            frame.after(200, step)

        game = Game(const.BOARD_SIZE)
        state = deque(maxlen = 4)
        root = tkinter.Tk()
        size = const.BOARD_SIZE * 30
        frame = tkinter.Frame(master = root, height = size, width = size)
        frame.pack()
        painter = Painter(frame, game.snake, game.grid)
        frame.after(0, start)
        root.mainloop()



