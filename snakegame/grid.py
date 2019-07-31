class Grid:
    def __init__(self, cell_number, cell_size):
         #just so size is multiple of cell_size
        self.cell_size = cell_size
        self.cell_number = cell_number 
        self.center = (self.cell_number//2, self.cell_number//2)
        self.color = "black"
        self.cell_cords = [(a, b) for a in range(self.cell_number) for b in range(self.cell_number)]
        self.reward_pos = None
        
    def reset(self):
        self.reward_pos = None

    def get_reward(self):
        return self.reward_pos

    def get_size(self):
        return self.cell_number

    def get_center(self):
        return self.center
