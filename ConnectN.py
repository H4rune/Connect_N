import numpy as np
import math
import matplotlib.pyplot as plt
import copy

class ConnectN:
    def __init__(self, size=(6, 7), connect=4):
        """
        Initialize the ConnectN environment.

        Args:
            size (tuple): A tuple (height, width) specifying the board size.
            connect (int): Number of consecutive pieces needed to win.
        """
        self.height, self.width = size
        self.connect = connect
        self.board_state = np.zeros((self.height, self.width), dtype=int)
        self.current_player = 1

    def reset(self, board=None):
        """
        Reset the board to the starting state. Optionally, set a provided board state.

        Args:
            board (list of lists, optional): Custom board state.
        """
        if board is None:
            self.board_state = np.zeros((self.height, self.width), dtype=int)
        else:
            arr = np.array(board)
            if arr.shape != (self.height, self.width):
                raise ValueError("Provided board shape does not match environment dimensions.")
            self.board_state = arr.copy()
        self.current_player = 1

    def is_valid_game_state(self, board):
        """
        Check if a given board state (list of lists) is valid.
        Validity means:
          - The board has the correct dimensions.
          - All cell values are 0, 1, or 2.
          - Gravity is respected (i.e. in each column, once an empty cell is found from the bottom, no cell above is filled).

        Args:
            board (list of lists): Board state to validate.

        Returns:
            bool: True if valid, False otherwise.
        """
        arr = np.array(board)
        if arr.shape != (self.height, self.width):
            return False
        if not np.all(np.isin(arr, [0, 1, 2])):
            return False
        for col in range(self.width):
            col_vals = arr[:, col]
            found_zero = False
            # Iterate from bottom row upward
            for val in col_vals[::-1]:
                if val == 0:
                    found_zero = True
                elif found_zero and val != 0:
                    return False
        return True

    def is_game_over(self, board=None):
        """
        Check the board for a win or draw condition.
        Returns:
          - 1 if player 1 wins.
          - 2 if player 2 wins.
          - 0 if the board is full (draw).
          - None if the game is not over.

        Args:
            board (list of lists or numpy array, optional): Board state to check.
                If None, uses the current board_state.

        Returns:
            int or None: Outcome of the game.
        """
        if board is None:
            board = self.board_state
        board = np.array(board)
        for i in range(self.height):
            for j in range(self.width):
                if board[i, j] != 0:
                    player = board[i, j]
                    # Check horizontal (to the right)
                    if j <= self.width - self.connect and np.all(board[i, j:j+self.connect] == player):
                        return player
                    # Check vertical (downwards)
                    if i <= self.height - self.connect and np.all(board[i:i+self.connect, j] == player):
                        return player
                    # Check diagonal down-right
                    if i <= self.height - self.connect and j <= self.width - self.connect:
                        if all(board[i+k, j+k] == player for k in range(self.connect)):
                            return player
                    # Check diagonal up-right
                    if i >= self.connect - 1 and j <= self.width - self.connect:
                        if all(board[i-k, j+k] == player for k in range(self.connect)):
                            return player
        # If the board is full, return draw (0)
        if not np.any(board == 0):
            return 0
        return None

    def get_number_of_actions(self):
        """
        Get the number of valid actions (columns not full).

        Returns:
            int: Count of columns where a chip can be played.
        """
        valid = 0
        for col in range(self.width):
            if self.board_state[0, col] == 0:
                valid += 1
        return valid

    def get_number_of_states(self):
        """
        Compute the number of possible states for the board.
        (Each cell can be in 3 states: 0, 1, or 2.)

        Returns:
            int: Total number of states, capped at 1,000,000.
        """
        total = 3 ** (self.height * self.width)
        if total > 1e6:
            return int(1e6)
        else:
            return total

    def get_state(self):
        """
        Get a copy of the current board state.

        Returns:
            numpy.array: The board state.
        """
        return self.board_state.copy()

    def get_reward(self, last_player):
        """
        Get the reward after the last move.
        Reward scheme:
          - 1 if the last move wins the game.
          - 0 for draw or non-terminal state.
          - -1 if somehow the last move resulted in a loss (should not happen in proper play).

        Args:
            last_player (int): The player who just made a move.

        Returns:
            int: Reward value.
        """
        outcome = self.is_game_over()
        if outcome is None:
            return 0
        elif outcome == 0:
            return 0  # draw
        elif outcome == last_player:
            return 1
        else:
            return -1

    def get_terminal_flag(self):
        """
        Check whether the current state is terminal.

        Returns:
            bool: True if game is over, False otherwise.
        """
        return self.is_game_over() is not None

    def execute_action(self, action):
        """
        Execute an action (choose a column) for the current player.
        Chips will "fall" to the bottom of the chosen column.
        Returns a tuple of (state, reward, terminal_flag).

        Args:
            action (int): Column index where to drop the chip.

        Returns:
            tuple: (board state, reward, terminal flag).

        Raises:
            ValueError: If the action is out of range or if the column is full.
        """
        if action < 0 or action >= self.width:
            raise ValueError("Action out of range.")
        placed = False
        for row in range(self.height - 1, -1, -1):
            if self.board_state[row, action] == 0:
                self.board_state[row, action] = self.current_player
                placed = True
                break
        if not placed:
            raise ValueError("Column is full.")
        last_player = self.current_player
        terminal = self.get_terminal_flag()
        reward = self.get_reward(last_player)
        # Switch player if game is not over.
        if not terminal:
            self.current_player = 2 if self.current_player == 1 else 1
        return self.get_state(), reward, terminal

    # def display_board(self, pretty=True, board=None):
    #     """
    #     Display the board. If pretty is True, generate an image using matplotlib;
    #     otherwise, print the board to the terminal.

    #     Args:
    #         pretty (bool, optional): Whether to display a pretty image. Defaults to True.
    #         board (list of lists or numpy array, optional): Board state to display.
    #             If None, uses the current board_state.
    #     """
    #     if board is None:
    #         board = self.board_state
    #     board = np.array(board)
    #     if pretty:
    #         fig, ax = plt.subplots()
    #         ax.set_xticks(np.arange(self.width))
    #         ax.set_yticks(np.arange(self.height))
    #         ax.set_xticklabels([])
    #         ax.set_yticklabels([])
    #         ax.grid(True)
    #         for i in range(self.height):
    #             for j in range(self.width):
    #                 if board[i, j] == 1:
    #                     ax.text(j, i, 'X', ha='center', va='center', color='red', fontsize=20)
    #                 elif board[i, j] == 2:
    #                     ax.text(j, i, 'O', ha='center', va='center', color='blue', fontsize=20)
    #         ax.invert_yaxis()  # so the bottom row appears at the bottom
    #         plt.title("ConnectN Board")
    #         plt.show()
    #     else:
    #         print(board)

    def display_board(self, pretty=True, board=None):
        """
        Display the board. If pretty is True, generate an image using matplotlib;
        otherwise, print the board to the terminal.

        Args:
            pretty (bool, optional): Whether to display a pretty image. Defaults to True.
            board (list of lists or numpy array, optional): Board state to display.
                If None, uses the current board_state.
        """
        if board is None:
            board = self.board_state
        board = np.array(board)
        if pretty:
            fig, ax = plt.subplots()
            # Set ticks to create a grid that outlines each cell.
            ax.set_xticks(np.arange(self.width + 1))
            ax.set_yticks(np.arange(self.height + 1))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.grid(True)
            # Place chips centered within each square.
            for i in range(self.height):
                for j in range(self.width):
                    if board[i, j] == 1:
                        ax.text(j + 0.5, i + 0.5, 'X', ha='center', va='center', color='red', fontsize=20)
                    elif board[i, j] == 2:
                        ax.text(j + 0.5, i + 0.5, 'O', ha='center', va='center', color='blue', fontsize=20)
            ax.invert_yaxis()  # so the bottom row appears at the bottom
            plt.title("ConnectN Board")
            plt.show()
        else:
            print(board)


    def play_game(self):
        """
        Allows two players to play the game manually in the terminal.
        """
        self.reset()
        while True:
            self.display_board(pretty=False)
            try:
                action = int(input(f"Player {self.current_player}, choose a column (0-{self.width-1}): "))
            except ValueError:
                print("Please enter a valid integer.")
                continue
            try:
                state, reward, terminal = self.execute_action(action)
            except ValueError as e:
                print(e)
                continue
            if terminal:
                self.display_board(pretty=False)
                outcome = self.is_game_over()
                if outcome == 0:
                    print("Game ended in a draw.")
                else:
                    print(f"Player {outcome} wins!")
                break

# -------------------------
# Unit Tests for ConnectN
# -------------------------
import unittest
import io
import sys

class TestConnectN(unittest.TestCase):
    def setUp(self):
        self.env = ConnectN(size=(6, 7), connect=4)
        self.env.reset()

    def test_reset(self):
        # Modify board then reset and check if board is all zeros.
        self.env.board_state[0, 0] = 1
        self.env.reset()
        self.assertTrue(np.all(self.env.board_state == 0))

        # Test reset with a provided board state.
        board = [[0] * 7 for _ in range(6)]
        board[5][0] = 1
        self.env.reset(board)
        self.assertEqual(self.env.board_state[5, 0], 1)

    def test_execute_action(self):
        # Execute an action in column 3.
        state, reward, terminal = self.env.execute_action(3)
        # The bottom cell in column 3 should now be filled with player 1's chip.
        self.assertEqual(state[5, 3], 1)
        self.assertEqual(reward, 0)  # game not over
        self.assertFalse(terminal)
        # Check that the current player has switched.
        self.assertEqual(self.env.current_player, 2)

        # Fill a column and then try an invalid move.
        self.env.reset()
        for _ in range(6):
            self.env.execute_action(0)
        with self.assertRaises(ValueError):
            self.env.execute_action(0)

    def test_is_valid_game_state(self):
        # Valid board state: chip at bottom of column 0.
        valid_board = [[0] * 7 for _ in range(6)]
        valid_board[5][0] = 1
        self.assertTrue(self.env.is_valid_game_state(valid_board))
        # Invalid board state: chip floating in column (empty cell below).
        invalid_board = [[0] * 7 for _ in range(6)]
        invalid_board[2][0] = 1  # not at the bottom and there are empty cells below
        self.assertFalse(self.env.is_valid_game_state(invalid_board))

    def test_is_game_over_horizontal(self):
        # Horizontal win for player 1 on bottom row.
        board = np.zeros((6, 7), dtype=int)
        board[5, 0:4] = 1
        self.env.reset(board.tolist())
        outcome = self.env.is_game_over()
        self.assertEqual(outcome, 1)

    def test_is_game_over_vertical(self):
        # Vertical win for player 2 in column 3.
        board = np.zeros((6, 7), dtype=int)
        board[5, 3] = 2
        board[4, 3] = 2
        board[3, 3] = 2
        board[2, 3] = 2
        self.env.reset(board.tolist())
        outcome = self.env.is_game_over()
        self.assertEqual(outcome, 2)

    def test_is_game_over_diagonal_down_right(self):
        # Diagonal down-right win for player 1.
        board = np.zeros((6, 7), dtype=int)
        board[2, 0] = 1
        board[3, 1] = 1
        board[4, 2] = 1
        board[5, 3] = 1
        self.env.reset(board.tolist())
        outcome = self.env.is_game_over()
        self.assertEqual(outcome, 1)

    def test_is_game_over_diagonal_up_right(self):
        # Diagonal up-right win for player 2.
        board = np.zeros((6, 7), dtype=int)
        board[5, 0] = 2
        board[4, 1] = 2
        board[3, 2] = 2
        board[2, 3] = 2
        self.env.reset(board.tolist())
        outcome = self.env.is_game_over()
        self.assertEqual(outcome, 2)

    def test_get_number_of_actions(self):
        # At the start, all 7 columns should be available.
        self.assertEqual(self.env.get_number_of_actions(), 7)
        # Fill column 0 completely.
        for _ in range(6):
            self.env.execute_action(0)
        self.assertEqual(self.env.get_number_of_actions(), 6)

    def test_get_number_of_states(self):
        # For a 6x7 board, total states = 3^(42) exceeds 1e6, so should return 1e6.
        self.assertEqual(self.env.get_number_of_states(), 1000000)
        # For a smaller board (2x2), total states = 3^4 = 81.
        small_env = ConnectN(size=(2, 2), connect=2)
        self.assertEqual(small_env.get_number_of_states(), 81)

    def test_get_state(self):
        # After a reset, get_state() should return the current board state.
        state = self.env.get_state()
        np.testing.assert_array_equal(state, self.env.board_state)

    def test_get_reward_and_terminal_flag(self):
        # Simulate a winning move (horizontal win for player 1).
        board = np.zeros((6, 7), dtype=int)
        board[5, 0:3] = 1
        self.env.reset(board.tolist())
        self.env.current_player = 1
        state, reward, terminal = self.env.execute_action(3)
        self.assertEqual(reward, 1)
        self.assertTrue(terminal)

    def test_display_board(self):
        # Test that display_board (non-pretty mode) prints output.
        captured_output = io.StringIO()
        sys.stdout = captured_output
        self.env.display_board(pretty=False)
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        self.assertIn("0", output)  # The printed board should contain zeros

if __name__ == '__main__':
    # Uncomment the next two lines to run unit tests
    #unittest.main()

    # Uncomment the lines below to play a manual game.
    env = ConnectN(size=(6, 7), connect=4)
    # env.play_game()
    env.execute_action(3)
    env.execute_action(4)
    env.execute_action(3)
    env.execute_action(4)
    env.execute_action(3)
    env.display_board(pretty=True)
    