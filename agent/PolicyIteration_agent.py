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

    def policy_evaluation(self): 
        """
        bước evaluation : 
            - tính hàm value state 
        """        

        while True : 
            delta = 0 
            for state in range(self.env.observation_space.n): 
                # lấy value và action của state hiện tại 
                v = self.value_state[state]
                action = self.policy[state]
                next_state, reward, done, _ = self.env.step(action)

                # cập nhật value state 
                self.value_state[state] = reward + self.gamma * self.value_state[next_state]
                delta = max(delta, abs(v - self.value_state[state]))
            if delta < self.epsilon:
                break
        return self.value_state

    def policy_improvement(self): 
        """
        Bước imporve policy : 
            - với mỗi hàm value state được tính -> cập nhật giá trị cho các bước 
        """
        policy_stable = True
        for state in range(self.env.observation_space.n): 
            old_action = self.policy[state]

            # duyệt qua action để lấy ra action tốt nhất 
            for action in range(self.env.action_space.n): 
                next_state, reward, done, _ = self.env.step(action)
                v = reward + self.gamma * self.value_state[next_state]
                self.policy[state] = np.argmax(v)
            if old_action != self.policy[state]:
                policy_stable = False

        return policy_stable



    def train(self, max_iteration= 1000):
        """
        Hàm huấn luyện : sử dụng thuật toán cho tới khi đại hiều kiện hội tụ 
        """
        count = 0 
        while count < max_iteration: 
            self.policy_evaluation()
            policy_stable = self.policy_improvement()
            count += 1 
            if policy_stable:
                break