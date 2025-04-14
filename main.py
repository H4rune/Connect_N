import numpy as np
import time
import copy
from matplotlib import pyplot as plt

from Project3_agentQ import QAgent
from Project3_agentS import SARSA_0
from Project3_env import GoldExplorer


# This is the training loop for the Q agent
def calculate_G(rewards_list: list, gamma: float):
    #  This method will return the episode return given
    #  a complete sequential list of all rewards in the episode
    # TODO: This Method is incorrect and needs to be rewritten

    G = 0
    # Reverse the list and start with most recent rewards
    rewards_list.reverse()
    for i, reward in enumerate(rewards_list):
        G = reward + gamma * G
    return G


def example_training_loop():
    agent = QAgent(epsilon=.2)
    env = GoldExplorer()
    # Iterate over the number of episodes
    for i in range(1000):
        # reset the game and observe the current state
        current_state = env.reset()
        game_end = False
        # Do until the game ends:
        while not game_end:
            # Obtain selected action
            selected_action = agent.select_action(current_state)
            # Execute action + Observe new state and reward
            new_state, reward, game_end = env.execute_action(selected_action)
            # Update q value (state, selected_action, reward, new_state)
            agent.update_q(current_state, selected_action, reward, new_state)
            current_state = new_state

        # update q table when game ends
        agent.update_q(i)


def Q_game_loop(agent, env, exp_starts=False):
    all_rewards = []

    # reset the game and observe the current state
    current_state = env.reset(exp_starts=exp_starts)
    game_end = False

    # Do until the game ends:
    while not game_end:
        # Obtain selected action
        selected_action = agent.select_action(current_state)
        # Execute action + Observe new state and reward
        new_state, reward, game_end = env.execute_action(selected_action)
        # Update q value (state, selected_action, reward, new_state)
        agent.update_q(current_state, selected_action, reward, new_state)
        # Store reward for G calculation later
        all_rewards.append(reward)
        # Progress to next state
        current_state = new_state

    return all_rewards

def S_game_loop(agent, env, exp_starts=False):
    all_rewards = []

    # reset the game and observe the current state
    current_state = env.reset(exp_starts=exp_starts)
    game_end = False

    # Obtain selected action
    selected_action = agent.select_action(current_state)

    # Do until the game ends:
    while not game_end:
        # Execute action + Observe new state and reward
        new_state, reward, game_end = env.execute_action(selected_action)
        # Obtain selected action
        next_action = agent.select_action(new_state)
        # Store reward for G calculation later
        all_rewards.append(reward)
        # Update q value (state, selected_action, reward, new_state)
        agent.update_q(current_state, selected_action, reward, new_state, next_action)

        # Progress to next state
        current_state = new_state
        selected_action = next_action

    return all_rewards


def collect_data(starting_agent, num_agents: int, episodes: int, e_max=1, e_min=.01, N=None, exp_starts=False):
    if N == None:
        N = episodes / 2

    trained_agents = []
    # G_full contains the rewards for each agent (rows) for every episode (columns)
    G_full = []
    for i in range(num_agents):
        agent = copy.deepcopy(starting_agent)
        agent.epsilon = e_max

        env = GoldExplorer()
        G_i = []
        for j in range(episodes):

            r = max(0, ((N - j) / N))
            agent.epsilon = (e_max - e_min) * r + e_min

            if agent.name == "Q":
                all_rewards = Q_game_loop(agent, env, exp_starts)
            else:
                all_rewards = S_game_loop(agent, env, exp_starts)

            # append G to G_i
            G_i.append(calculate_G(all_rewards, agent.gamma))
        G_full.append(copy.deepcopy(G_i))
        trained_agents.append(copy.deepcopy(agent))
    return G_full, trained_agents


def obtain_stats(G):
    nparr = np.array(G)
    G_sem = (np.std(nparr, axis=0) / np.sqrt(len(G))).tolist()
    G_avg = np.mean(nparr, axis=0).tolist()

    return G_avg, G_sem


def plot_data(G: list, agent_name: str):
    avg, sem = obtain_stats(G)

    plt.figure()
    plt.title(f"{agent_name} Performance")
    plt.xlabel("Episode Number")
    plt.ylabel("Average Return")
    plt.plot(avg)
    plt.savefig(fname=f"{agent_name}_full_p")

    plt.figure()
    plt.title(f"{agent_name} Standard Error")
    plt.xlabel("Episode Number")
    plt.ylabel("Average Standard Error")
    plt.plot(sem)
    plt.savefig(fname=f"{agent_name}_full_se")
    plt.close()

    avg_last50 = np.average(avg[-50:])
    sem_last50 = np.average(sem[-50:])

    headers = ["Return", "Standard Error"]
    data = [[f"{avg_last50:.4f}", f"{sem_last50:.4f}"]]

    fig, ax = plt.subplots(figsize=(2, 2), dpi=200)
    plt.title(f"{agent_name} Averages")
    ax.axis("tight")
    ax.axis("off")
    ax.table(cellText=data, colLabels=headers, cellLoc="center", loc="center")
    plt.savefig(fname=f"{agent_name}_last50")
    plt.close()


def Q_agent_training_without_expstarts():
    agent = QAgent(alpha=.1, gamma=.95, epsilon=.8)
    env = GoldExplorer()
    start = time.time()
    G, agents = collect_data(agent, 50, 1000, 1, .01, 900, False)
    end = time.time()
    # print(end - start)
    plot_data(G, "Q_agent_1")

def Q_agent_training_with_expstarts():
    agent = QAgent(alpha=.4, gamma=.95, epsilon=.8)
    env = GoldExplorer()
    start = time.time()
    G1, agents = collect_data(agent, 50, 500, 1, .3, 500, True)
    agents[0].alpha = .1
    G2, agents = collect_data(agents[0], 50, 200, .01, .01, 50, False)
    end = time.time()
    # print(end - start)
    plot_data(G1, "Q_agent_2_exp")
    plot_data(G2, "Q_agent_2")

def S_agent_training_without_expstarts():
    agent = SARSA_0(alpha=.1, gamma=.95, epsilon=.8)
    env = GoldExplorer()
    start = time.time()
    G, agents = collect_data(agent, 50, 1000, 1, .01, 950, False)
    end = time.time()
    # print(end-start)
    plot_data(G, "SARSA_agent_1")

def S_agent_training_with_expstarts():
    agent = SARSA_0(alpha=.4, gamma=.95, epsilon=.8)
    env = GoldExplorer()
    start = time.time()
    G1, agents = collect_data(agent, 50, 500, 1, .3, 500, True)
    agents[0].alpha = .1
    G2, agents = collect_data(agents[0], 50, 200, .01, .01, 50, False)
    end = time.time()
    # print(end-start)
    plot_data(G1, "SARSA_agent_2_exp")
    plot_data(G2, "SARSA_agent_2")

def main():
    Q_agent_training_with_expstarts()
    print("Q agent with exploring starts complete")
    Q_agent_training_without_expstarts()
    print("Q agent without exploring starts complete")

if __name__ == "__main__":
    main()