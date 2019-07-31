from tkinter import *
import random as rnd

TIMESTEP = 200
HEIGHT = WIDTH = 200

class Snake:
    def __init__(self, start):
        self.going = 'Up'
        self.heading_changed = False
        self.head = start
        self.frozen = False
        self.points = 0
        self.eaten = 0
        self.color = "blue"
        self.to_grow = False
        self.body = [self.head]
        self.alive = True
        self.food_dist = 5

    def move(self):
        self.heading_changed = False
        if self.going == 'Up':
            newhead = (self.head[0] - 1, self.head[1])
        elif self.going == 'Right':
            newhead = (self.head[0], self.head[1] + 1)
        elif self.going == 'Down':
            newhead = (self.head[0] + 1, self.head[1])
        else:
            newhead = (self.head[0], self.head[1] - 1)
        self.head = newhead
        if self.to_grow:
            self.body = [newhead] + self.body
            self.to_grow = False
        else:
            self.body = [newhead] + self.body[:-1]


    def grow(self):
        self.to_grow = True
        self.eaten += 1

    def freeze(self):
        self.frozen = not self.frozen

    def hit_the_wall(self, n):
        if self.head[0] in [-1, n] or self.head[1] in [-1,n]:
            return True
        return False

    def bit_itself(self):
        return self.head in self.body[1:]

    def die(self):
        self.frozen = True
        self.alive  = False

    def set_heading(self, go):
        if self.frozen or self.heading_changed:
            return
        forbidden = zip(['Up', 'Right', 'Left', 'Down'], ['Up', 'Right', 'Left', 'Down'][::-1])
        if (self.going, go) not in forbidden:
            self.going = go
            self.heading_changed = True

    
    def on_reward(self, reward):
        return self.head == reward

    def add_pts(self, n):
        self.points += n

    def get_head(self):
        return self.head   

    def reset(self, start):
        self.going = 'Up'
        self.heading_changed = False
        self.head = start
        self.frozen = False
        self.points = 0
        self.eaten = 0
        self.to_grow = False
        self.body = [self.head]
        self.alive = True








