import numpy as np 


class ValueIteration_Agent: 

    def __init__(self, env,  gamma: float = 0.9, epsilon: float = 0.1):
        self.env = env 
        self.gamma = gamma 
        self.epsilon = epsilon

        self.policy = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.state_value = np.zeros(self.env.observation_space.n)
        self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))




    
    def value_iteration(self, max_iter = 1000):

        """
        Hàm cài đặt value iteration: 
            return : Q-table sau khi hội tụ
        """
        
        while True: 
            delta = 0 
            for state in range(self.env.observation_space.n):
                for action in range(self.env.action_space.n):
                    next_state, reward, done, _ = self.env.step(action)
                    self.Q[state][action] = reward + self.gamma * self.state_value[next_state]
                # lấy giá trị lớn nhất của action value
                v = np.max(self.Q[state])
                delta = max(delta, abs(v - self.state_value[state]))
                self.state_value[state] = v

            if delta < self.epsilon:
                break

        self.update_policy()
        
    

    def update_policy(self):
        """
        Cập nhật policy sau khi value đã hội tụ
        """
        for state in range(self.env.observation_space.n):
            self.policy[state] = np.argmax(self.Q[state])
        return self.policy



    def train(self, max_iteration = 1000):
        """
        Huân luyện với số lượng iteration max 
        """

        self.value_iteration(max_iteration)   
