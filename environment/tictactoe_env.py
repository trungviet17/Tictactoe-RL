import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TicTacToeEnv(gym.Env):
    def __init__(self, size: int = 3):
        super(TicTacToeEnv, self).__init__()
        self.size = size 
        self.action_space = spaces.Discrete(self.size**2)  
        self.observation_space = spaces.Box(low=0, high=2, shape=(self.size, self.size), dtype=np.int8) 
        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=np.int8) 
        self.done = False
        return self.board.flatten()

    def step(self, action):
        # kiểm tra trùng lặp 
        if self.board[action // self.size, action % self.size ] != 0 or self.done:
            return self.board.flatten(), -10, True, {}  

        
        self.board[action // self.size, action % self.size] = 1  

        # kiểm tra ngừoi chơi 1 có thắng 
        if self.check_winner(1):
            return self.board.flatten(), 1, True, {}  # người chơi X thắng, +1 điểm

        #
        if not 0 in self.board.flatten():
            return self.board.flatten(), 0, True, {}  # hòa
        # lựa chọn bước đi ngẫu nhiên 
        opponent_action = np.random.choice(np.where(self.board.flatten() == 0)[0])
        self.board[opponent_action // self.size, opponent_action % self.size] = 2  # người chơi O đi

        if self.check_winner(2):
            return self.board.flatten(), -1, True, {} # thua trừ 2 
        return self.board.flatten(), 0, False, {}


    def check_winner(self, player):
        for i in range(self.size):
            if all([self.board[i, j] == player for j in range(self.size)]) or all([self.board[j, i] == player for j in range(self.size)]):
                return True
        
        if all([self.board[i, i] ==  player for i in range(self.size)])  or all([self.board[self.size - i - 1, i] ==  player for i in range(self.size)]):
            return True
        return False
