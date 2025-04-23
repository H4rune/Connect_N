import numpy as np
import random

class QAgent:
    def __init__(self, env, epsilon=1.0, alpha=0.1, gamma=0.99, *args, **kwargs):
        """
        Initialize the Q-learning agent for the ConnectN environment.
        
        Parameters:
        - env: An instance of the ConnectN environment.
        - epsilon: Exploration probability.
        - alpha: Learning rate.
        - gamma: Discount factor.
        """
        self.env = env
        self.num_states = self.env.get_number_of_states()
        self.num_actions = self.env.width
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = np.zeros((self.num_states, self.num_actions))
        

    def _state_to_index(self, board):
        """
        Vectorised base‑3 hash of the board.
        """
        flat = board.flatten()
        # Pre‑compute 3^k vector once and cache it
        if not hasattr(self, "_pow3"):
            self._pow3 = (3 ** np.arange(flat.size, dtype=np.int64)) % self.num_states
        idx = int((flat.astype(np.int64) * self._pow3).sum() % self.num_states)
        return idx


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
