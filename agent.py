import random
import tkinter
import torch
import torch.optim as optim

import structures.const as const
from structures.network import DQN
from structures.replay_memory import ReplayMemory

from snakegame.game import Game
from snakegame.painter import Painter

class Agent:
    def __init__(self):
        self.net = DQN()
        self.net.cuda()
        self.memory = ReplayMemory(10000)
        self.optimizer = optim.Adam(self.net.parameters())

        self.env = Game(const.BOARD_SIZE)
        self.state = self.env.get_state()
        self.iter = 0

    def train(self):
        self.net.train()
        while True:
            action = self.net.eps_greedy(self.state, self.iter) 
            _dict = self.env.step(action)
            reward = torch.tensor(_dict['reward'], dtype = torch.float)
            dead = torch.tensor(_dict['dead'], dtype = torch.float)
            next_state = self.env.get_state()
            self.memory.push(self.state, action, reward, next_state, dead)
            self.state = next_state
            
            if len(self.memory) >= const.BATCH_SIZE: 
                self._perform_gradient_step()
            self.iter += 1
            print(self.iter)
            if dead:
                self.env.newgame()

    def _perform_gradient_step(self):
        states, actions, rewards, next_states, deaths = self.memory.sample()
        with self.net.LOCK:
            q_vals = self.net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_vals = self.net(next_states).max(1)[0]

        expected_q_value = rewards + const.GAMMA * next_q_vals * (1 - deaths)
        loss = (q_vals - expected_q_value).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def visible_play(self):
        def start():
            game.newgame()
            painter.drawnewgame(game.snake)
            frame.after(0, step)
            
        def step():
            action = self.net.greedy(game.get_state().cuda())
            _dict = game.step(action)
            if _dict['dead']:
                frame.after(3000, start)
                return
            painter.draw_changes(game.grid.get_reward())
            frame.after(200, step)

        game = Game(const.BOARD_SIZE)
        root = tkinter.Tk()
        size = const.BOARD_SIZE * 30
        frame = tkinter.Frame(master = root, height = size, width = size)
        frame.pack()
        painter = Painter(frame, game.snake, game.grid)
        frame.after(0, start)
        root.mainloop()


