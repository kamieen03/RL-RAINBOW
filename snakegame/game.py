import random as rnd
import numpy as np
import torch

from snakegame.snake import Snake
from snakegame.grid import Grid

class Game:
    def __init__(self, board_size, cell_size = 30):
        self.grid = Grid(board_size, cell_size)
        self.snake = Snake(self.grid.get_center())

    def newgame(self):
        self.grid.reset()
        self.snake.reset(self.grid.center)
        self.place_new_reward()

    def place_new_reward(self):
        self.grid.reward_pos = rnd.choice(list(set(self.grid.cell_cords) - set(self.snake.body)))

    def play(self):
        if not self.snake.frozen:
            self.snake.move()
            if self.snake.hit_the_wall(self.grid.get_size()) or self.snake.bit_itself():
                self.snake.die()
                return 'DEAD'
            elif self.snake.on_reward(self.grid.get_reward()):
                self.snake.grow()
                self.snake.add_pts(20.0)
                self.place_new_reward()
                return 20
            else:
                return 0

    def step(self, action):
        """
        RL agent interface.
        :param action:
        :return:
        """
        action = self.get_actions()[action]
        self.snake.set_heading(action)
        self.snake.move()

        reward = 0; dead = False
        if self.snake.hit_the_wall(self.grid.get_size()) or self.snake.bit_itself():
            self.snake.die()
            dead = True
            reward = -1.0
        elif self.snake.on_reward(self.grid.get_reward()):
            self.snake.grow()
            self.place_new_reward()
            reward = 1.0
        else:
            reward = -0.02
        return {'dead': dead,
                'reward': reward}

    def get_state(self):
        if not self.snake.alive:
            state = -np.ones((self.grid.cell_number, self.grid.cell_number))
        else:
            body = self.snake.body
            head = body[0]
            food = self.grid.get_reward()

            state = np.zeros((self.grid.cell_number, self.grid.cell_number))
            for c in body[1:]: state[c[0], c[1]] = 0.33
            state[head[0], head[1]] = 0.66
            if food is not None:
                state[food[0], food[1]] = 1.0
        return torch.tensor(state, dtype = torch.float).unsqueeze(0).unsqueeze(0).cuda()

    def get_actions(self):
        return ['Up', 'Right', 'Down', 'Left']
