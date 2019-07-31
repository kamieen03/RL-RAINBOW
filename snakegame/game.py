import random as rnd

from snakegame.snake import Snake
from snakegame.grid import Grid

class Game:
    def __init__(self, board_size ,cell_size):
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

    def step(action):
        self.snake.set_heading(action)
        self.snake.move()

        reward = 0; dead = False
        if self.snake.hit_the_wall(self.grid.get_size()) or self.snake.bit_itself():
            self.snake.die()
            dead = True
            reward = -100
        elif self.snake.on_reward(self.grid.get_reward()):
            self.snake.grow()
            self.snake.add_pts(20.0)
            self.place_new_reward()
            reward = 20
        return {'dead': dead,
                'reward': reward}

    def get_state():
        #TODO
        pass
