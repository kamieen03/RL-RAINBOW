import tkinter

class Painter:
    def __init__(self, master, snake, grid):
        self.master = master
        self.grid = grid
        self.cells = {}
        self.snake = snake
        self.old_snake_body = snake.body.copy()
        self.rew_pos = grid.get_reward()
        self.snake_c = snake.color
        self.grid_c = grid.color
        self.head_c = 'green'
        self.food_c = "red"
        for cell in grid.cell_cords:
            c = tkinter.Frame(master = master, height = self.grid.cell_size, width = self.grid.cell_size, bg = self.grid_c)
            c.grid(row = cell[0], column = cell[1])
            self.cells[cell] = c
        

    def draw_changes(self, food):
        # color head
        self.turn_color(self.snake.head, self.head_c)

        #change color of part that was head in previous frame
        if len(self.snake.body) > 1:
            self.turn_color(self.snake.body[1], self.snake_c)

        if self.old_snake[-1] not in self.snake.body:
            self.turn_color(self.old_snake[-1], self.grid_c)
        if food != self.rew_pos:
            self.turn_color(self.rew_pos, self.head_c)
            self.turn_color(food, self.food_c)
            self.rew_pos = food
        self.old_snake = self.snake.body[:]

    def drawnewgame(self, snake):
        self.snake = snake
        self.rew_pos = self.grid.get_reward()
        self.old_snake = snake.body.copy()

        for c in self.grid.cell_cords:
            self.turn_color(c, self.grid_c)
        self.turn_color(self.rew_pos, self.food_c)
        self.turn_color(self.snake.head, self.head_c)


    def turn_color(self, pos, color):
        self.cells[pos].config(bg = color)

