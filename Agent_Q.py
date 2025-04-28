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
        legal = [c for c in range(self.num_actions) if self.env.board_state[0, c] == 0]
        if random.random() < self.epsilon:
             return random.choice(legal)
        else:
            q_legal = self.q_table[state_index, legal] 
            max_q = np.max(q_legal) #self.q_table[state_index]
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

    def greedy_action(self, state: np.ndarray) -> int:
        """
        Pick the best legal action for the given board state.
        Ties are broken uniformly at random.
        """
        # 1) Compute which columns are legal
        legal = [c for c in range(self.num_actions) if state[0, c] == 0]

        # 2) Compute the discrete index for this board
        idx = self._state_to_index(state)

        # 3) Gather Q-values for just those legal actions
        q_vals = self.q_table[idx, legal]

        # 4) Find the maximum Q among those
        max_q = np.max(q_vals)

        # 5) Collect all legal actions achieving that max, break ties
        best = [a for a, q in zip(legal, q_vals) if q == max_q]
        return random.choice(best)
