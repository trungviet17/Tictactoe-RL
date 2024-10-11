from environment.tictactoe_env import TicTacToeEnv
from agent.SARSA_agent import SARSA_Agent
import argparse


class Trainer: 
    """
    Class dùng để train và test giữa nhiều mô hình khác nhau 
    
    """
    def __init__(self, env):
        self.env = env



    def train(self, episode: int = 1000, model_name:str = 'SARSA'):
        """
        Setup số lượng episodes và tên mô hình cần train
        """
        if  model_name == 'SARSA':
            agent = SARSA_Agent(self.env)
            agent.train(episode)


         


    def test(self, model_name: str): 
        """
        Test policy của mô hình 
        """


        pass


    def plot_winrate(self): 
        """
        Vẽ win rate 
        """


        pass 




if __name__ == '__main__': 

    
    pass 