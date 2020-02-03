#!/usr/bin/env python3
import tkinter
from snakegame.app import App

def main():
    root = tkinter.Tk()
    app = App(root)
    app.start()
    root.mainloop()

if __name__ == '__main__':
    main()
