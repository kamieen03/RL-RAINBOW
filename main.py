#!/usr/bin/env python3
from threading import Thread

from agent import Agent

def main():
    agent = Agent()
#    th = Thread(target = agent.visible_play)
#    th.start()
    agent.train()

if __name__ == '__main__':
    main()
