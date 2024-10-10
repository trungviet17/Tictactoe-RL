import numpy as np 
import random 


class PolicyIteration_Agent: 

    def __init__(self, env, alpha = 0.1,  gamma=0.9, epsilon = 0.1):
        self.env = env 
        self.alpha = alpha
        self.gamma = gamma 
        self.epsilon = epsilon 

        self.policy = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.value_state = np.zeros(self.env.observation_space.n)


    def choose_action(self, state): 

        if random.uniform(0, 1) < self.epsilon: 
            return self.env.action_space.sample()
        else: 
            return self.better_action(state)



    def better_action(self, state):
        actions = range(self.env.action_space.n)
        q_values = [self.get_Q(state, a) for a in actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
        return random.choice(best_actions)


    def policy_evaluation(self): 
        """
        
        """        

        while True : 
            delta = 0 
            for state in range(self.env.observation_space.n): 
                v = self.value_state[state]
                action = self.policy[state]

                next_state, reward, done, _ = self.env.step(action)
                self.value_state[state] = reward + self.gamma * self.value_state[next_state]
                







    def policy_improvement(self): 
        pass



    def train(self): 
        pass 