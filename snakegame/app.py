import tkinter

from snakegame.painter import Painter
from snakegame.game import Game

import structures.const as const


BOARD_SIZE = const.BOARD_SIZE
CELL_SIZE = 30

class App:
    def __init__(self, root):
        self.root = root
        self.root.bind('<Key>', self.key_handler)
        self.frame = tkinter.Frame(master = root, height = BOARD_SIZE * CELL_SIZE, width = BOARD_SIZE * CELL_SIZE)
        self.frame.pack()
        self.game = Game(BOARD_SIZE, CELL_SIZE)
        self.painter = Painter(self.frame, self.game.snake, self.game.grid)
        self.key_pressed = False

    def start(self):
        self.game.newgame()
        self.painter.drawnewgame(self.game.snake)
        self.frame.after(0, self.step)
        
    def step(self):
        self.key_pressed = False
        state = self.game.play()
        if state == 'DEAD':
            return
        self.painter.draw_changes(self.game.grid.get_reward())
        self.frame.after(200, self.step)
        
    def key_handler(self, event):
        if self.key_pressed: return

        if event.keysym in ['Up', 'Right', 'Down', 'Left']:
            self.game.snake.set_heading(event.keysym)
        elif event.keysym == 'space' and self.game.snake.alive:
            self.game.snake.freeze()
        elif event.keysym in ['Escape', 'q']:
            event.widget.quit()
        elif event.keysym in ['n', 'space'] and not self.game.snake.alive:
            self.start()
        
        if self.game.snake.alive:
            self.key_pressed = True

