# ---------- main_new.py ----------
"""
Train & evaluate Q‑learning and DQN agents on Connect‑N.
Saves artefacts to ./models, plots to ./plots.
"""

import os, time
from typing import List, Dict, Tuple, Callable

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm

from ConnectN  import ConnectN
from Agent_Q   import QAgent
from Agent_DQN import DQNAgent

# ------------------------------------------------ utility -------------------
def calculate_G(rewards: List[int], gamma: float) -> float:
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
    return G

# ------------------------------------------------ episode loops -------------
def play_episode_q(agent: QAgent, env: ConnectN, gamma: float) -> Tuple[float,int,int]:
    env.reset()
    s_idx = agent._state_to_index(env.get_state())
    rewards, steps, done = [], 0, False
    while not done:
        a = agent.e_greedy(s_idx)
        try:
            nxt, r, done = env.execute_action(a)
        except ValueError:
            continue
        nxt_idx = agent._state_to_index(nxt)
        agent.update(s_idx, a, r, nxt_idx)
        rewards.append(r); s_idx = nxt_idx; steps += 1
    return calculate_G(rewards, gamma), rewards[-1], steps

def play_episode_dqn(agent: DQNAgent, env: ConnectN) -> Tuple[float,int,int]:
    env.reset(); state = env.get_state(); rewards, steps, done = [], 0, False
    while not done:
        while True:
            act = agent.select_action(state)
            try:
                nxt, r, done = env.execute_action(act); break
            except ValueError:
                continue
        agent.buffer.push(state, act, r, nxt, done); agent._optimize()
        rewards.append(r); state, steps = nxt, steps+1
    return calculate_G(rewards, agent.gamma), rewards[-1], steps

# ------------------------------------------------ metric collection ---------
def collect_metrics(agent_factory: Callable[[], object],
                    episodes: int, n_agents: int,
                    eps_max: float, eps_min: float,
                    decay_factor: float = 0.995) -> Tuple[Dict[str,List[List]], List[object]]:

    metrics = {k: [] for k in ("return","win","draw","len")}
    agents  = []

    outer = tqdm(range(n_agents), desc="Agents", unit="agent")
    for _ in outer:
        agent = agent_factory()                 # fresh instance
        env   = ConnectN()
        agent.epsilon = eps_max
        agents.append(agent)

        series = {k: [] for k in metrics}
        inner = tqdm(range(episodes), desc="Episodes", unit="ep", leave=False)
        for ep in inner:
            # simple exponential epsilon decay
            agent.epsilon = max(eps_min, agent.epsilon * decay_factor)

            if isinstance(agent, QAgent):
                G, last_r, length = play_episode_q(agent, env, agent.gamma)
            else:
                G, last_r, length = play_episode_dqn(agent, env)

            series["return"].append(G)
            series["win"].append(int(last_r == 1))
            series["draw"].append(int(last_r == 0))
            series["len"].append(length)

        for k in metrics:
            metrics[k].append(series[k])

    return metrics, agents

# ------------------------------------------------ plotting -------------------
def mean_sem(mat: List[List[float]]) -> Tuple[np.ndarray,np.ndarray]:
    arr = np.asarray(mat,dtype=float)
    return arr.mean(0), arr.std(0,ddof=1) / np.sqrt(len(arr))

def plot_series(metric, mean, sem, label, folder="plots"):
    os.makedirs(folder, exist_ok=True)
    plt.figure()
    plt.title(f"{label} – {metric}")
    plt.xlabel("Episode")
    plt.ylabel({"return":"Return","win":"Win‑rate",
                "draw":"Draw‑rate","len":"Moves"}[metric])
    plt.plot(mean)
    plt.fill_between(range(len(mean)), mean-sem, mean+sem, alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(folder,f"{label}_{metric}.png")); plt.close()

# ------------------------------------------------ evaluation (greedy) -------
# def evaluate(agent, episodes=100) -> Dict[str,float]:
#     env = ConnectN()
#     wins = draws = tot_len = 0
#     for _ in range(episodes):
#         env.reset(); state = env.get_state(); steps = 0; done=False
#         while not done:
#             if isinstance(agent, QAgent):
#                 a = np.argmax(agent.q_table[agent._state_to_index(state)])
#             else:
#                 valids = [c for c in range(env.width) if env.board_state[0,c]==0]
#                 with torch.no_grad():
#                     q = agent.policy_net(agent._state_tensor(state)).cpu().squeeze()
#                     q[[c for c in range(env.width) if c not in valids]] = -np.inf
#                     a = int(torch.argmax(q))
#             try:
#                 state,r,done = env.execute_action(a)
#             except ValueError:
#                 continue
#             steps += 1
#         wins  += (r==1)
#         draws += (r==0)
#         tot_len += steps
#     return {"win_rate":wins/episodes,"draw_rate":draws/episodes,
#             "avg_len":tot_len/episodes}

def evaluate(agent, episodes=100) -> Dict[str, float]:
    env = ConnectN()
    wins = draws = tot_len = 0
    H, W = env.height, env.width
    MAX_MOVES = H * W           # absolute safety cap

    for _ in range(episodes):
        env.reset()
        state = env.get_state()
        done = False
        steps = 0

        while not done and steps < MAX_MOVES:
            # ----- choose greedy but legal action -----
            valid_cols = [c for c in range(W) if env.board_state[0, c] == 0]
            if not valid_cols:            # board full -> draw
                break

            if isinstance(agent, QAgent):
                q_row = agent.q_table[agent._state_to_index(state)].copy()
                q_row[[c for c in range(W) if c not in valid_cols]] = -np.inf
                action = int(np.argmax(q_row))
            else:                         # DQNAgent
                with torch.no_grad():
                    q = agent.policy_net(agent._state_tensor(state)).cpu().squeeze()
                    q[[c for c in range(W) if c not in valid_cols]] = -np.inf
                    action = int(torch.argmax(q))

            # ----- execute -----
            try:
                state, r, done = env.execute_action(action)
            except ValueError:            # should not happen now
                continue

            steps += 1

        # ----- outcome bookkeeping -----
        if done:
            wins  += int(r == 1)
            draws += int(r == 0)
        else:                     # exceeded MAX_MOVES without terminal -> draw
            draws += 1
        tot_len += steps

    return {"win_rate": wins / episodes,
            "draw_rate": draws / episodes,
            "avg_len":   tot_len / episodes}

# ------------------------------------------------ experiment settings -------
N_AGENTS_Q    = 50
N_AGENTS_DQN  = 5
EPIS_Q        = 10000
EPIS_DQN      = 10000
EPS_MAX       = 1.0
EPS_MIN       = 0.05
DECAY_FACTOR  = 0.995

# ------------------------------------------------ runners -------------------
def run_q():
    factory = lambda: QAgent(env=ConnectN(), epsilon=EPS_MAX, alpha=.2, gamma=.95)
    m, agents = collect_metrics(factory, EPIS_Q, N_AGENTS_Q, EPS_MAX, EPS_MIN, DECAY_FACTOR)
    for k in m: plot_series(k,*mean_sem(m[k]),"QAgent")
    os.makedirs("models",exist_ok=True)
    np.save("models/QAgent_qtable.npy", agents[0].q_table)
    print("Q‑table saved.  Eval:", evaluate(agents[0]))

def run_dqn():
    factory = lambda: DQNAgent(env=ConnectN(), epsilon=EPS_MAX,
                               epsilon_decay=DECAY_FACTOR, buffer_cap=20000,
                               batch_size=128, lr=1e-3)
    m, agents = collect_metrics(factory, EPIS_DQN, N_AGENTS_DQN, EPS_MAX, EPS_MIN, DECAY_FACTOR)
    for k in m: plot_series(k,*mean_sem(m[k]),"DQNAgent")
    os.makedirs("models",exist_ok=True)
    torch.save(agents[0].policy_net.state_dict(),"models/DQNAgent_policy.pt")
    print("DQN weights saved. Eval:", evaluate(agents[0]))

# ------------------------------------------------ main ----------------------
if __name__ == "__main__":
    print(f'gpu: {torch.cuda.is_available()}')
    start = time.time()
    #run_q()
    run_dqn()
    print(f"Runtime: {time.time()-start:.1f}s")
