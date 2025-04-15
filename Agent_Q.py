import numpy as np
import random

class QAgent:
    def __init__(self, env, epsilon=1.0, alpha=0.1, gamma=0.99):
        """
        Initialize the Q-learning agent for the ConnectN environment.
        
        Parameters:
        - env: An instance of the ConnectN environment.
        - epsilon: Exploration probability.
        - alpha: Learning rate.
        - gamma: Discount factor.
        """
        self.env = env
        # Use the environment's provided function to compute the number of states.
        # The environment caps this number at 1,000,000 if necessary.
        self.num_states = self.env.get_number_of_states()
        # The number of actions equals the number of columns on the board.
        self.num_actions = self.env.width
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # Initialize the Q-table as a 2D array of zeros.
        self.q_table = np.zeros((self.num_states, self.num_actions))
        
    def _state_to_index(self, board):
        """
        Convert the current board state (numpy array) into a discrete state index.
        The board is treated as a base-3 number (each cell can be 0, 1, or 2)
        and then reduced modulo self.num_states to get an index within the Q-table.
        
        Parameters:
        - board: A numpy array representing the board state.
        
        Returns:
        - int: The discrete state index.
        """
        flat = board.flatten()
        index = 0
        base = 1
        for val in flat:
            index += int(val) * base
            base *= 3
        # Use modulo to ensure the index fits within the Q-table dimensions.
        return index % self.num_states

    def e_greedy(self, state_index):
        """
        Choose an action using an epsilon-greedy policy based on the current Q-table.
        
        Parameters:
        - state_index: The discrete index for the current state.
        
        Returns:
        - int: The chosen action (column index).
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            max_q = np.max(self.q_table[state_index])
            # In case multiple actions have the same Q-value, choose randomly among them.
            best_actions = [a for a in range(self.num_actions) if self.q_table[state_index, a] == max_q]
            return random.choice(best_actions)

    def update(self, state_index, action, reward, next_state_index):
        """
        Update the Q-table using the Q-learning update rule:
        
            Q(s,a) ← Q(s,a) + alpha * [reward + gamma * max_a' Q(next_state, a') - Q(s,a)]
        
        Parameters:
        - state_index: The discrete index for the current state.
        - action: The action taken.
        - reward: The reward received after the action.
        - next_state_index: The discrete index for the next state.
        """
        best_next_q = np.max(self.q_table[next_state_index])
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - self.q_table[state_index, action]
        self.q_table[state_index, action] += self.alpha * td_error

    def generate_episode(self):
        """
        Generate an episode by interacting with the ConnectN environment using the current policy.
        The agent updates its Q-values using the Q-learning rule after each step.
        
        Returns:
        - episode: A list of (state_index, action, reward) tuples encountered during the episode.
        """
        episode = []
        # Reset the environment; the environment's reset method does not return a state,
        # so we obtain the state via get_state().
        self.env.reset()
        state = self.env.get_state()
        state_index = self._state_to_index(state)
        done = False
        
        while not done:
            action = self.e_greedy(state_index)
            # Execute the chosen action.
            next_state, reward, done = self.env.execute_action(action)
            next_state_index = self._state_to_index(next_state)
            episode.append((state_index, action, reward))
            # Update Q-value for (state, action).
            self.update(state_index, action, reward, next_state_index)
            # Proceed to next state.
            state_index = next_state_index
        
        return episode

    def get_policy(self):
        """
        Extract the greedy policy from the current Q-table.
        For each discrete state index, this function returns the best action according to Q-values.
        
        Returns:
        - policy: A dictionary mapping state indices to their best action.
        """
        policy = {}
        for s in range(self.num_states):
            policy[s] = np.argmax(self.q_table[s])
        return policy
