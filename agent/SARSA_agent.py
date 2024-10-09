import numpy as np
import random

class SARSA_Agent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = {}  

    def get_Q(self, state, action):
        """Lấy giá trị Q của một trạng thái và hành động."""
        return self.Q.get((tuple(state), action), 0)

    def choose_action(self, state):
        """Chọn hành động theo epsilon-greedy."""
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.best_action(state)

    def best_action(self, state):
        """Chọn hành động có giá trị Q cao nhất."""
        actions = range(self.env.action_space.n)
        q_values = [self.get_Q(state, a) for a in actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
        return random.choice(best_actions)

    def update_Q(self, state, action, reward, next_state, next_action):
        """Cập nhật giá trị Q-table."""
        current_q = self.get_Q(state, action)
        next_q = self.get_Q(next_state, next_action)
        self.Q[(tuple(state), action)] = current_q + self.alpha * (reward + self.gamma * next_q - current_q)

    def train(self, episodes=10000):
        for episode in range(episodes):
            state = self.env.reset()
            action = self.choose_action(state)
            done = False

            while not done:
                next_state, reward, done, _ = self.env.step(action)
                next_action = self.choose_action(next_state)

                self.update_Q(state, action, reward, next_state, next_action)

                state = next_state
                action = next_action

            if (episode + 1) % 100000 == 0:
                print(f'Episode: {episode + 1}')

