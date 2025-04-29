import os, time, json, random
from typing import List, Dict, Tuple, Callable

import numpy as np
import pandas as pd                                 
import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm

from ConnectN   import ConnectN
from Agent_Q    import QAgent
from Agent_DQN  import DQNAgent

torch.backends.cudnn.benchmark = True


BOARD_SIZE, CONNECT  = (4, 4), 4
EPISODES_Q,  AGENTS_Q  = 30000 , 50
EPISODES_DQN, AGENTS_DQN = 30000 , 1


EPS_MAX, EPS_MIN, DECAY = 1.0, 0.05, 0.995
Q_PARAMS   = {"epsilon": EPS_MAX, "alpha": 0.1, "gamma": 0.95}
DQN_PARAMS = {"epsilon": EPS_MAX, "epsilon_decay": DECAY,
              "epsilon_min": EPS_MIN, "lr": 1e-3, "gamma": 0.95,
              "buffer_cap":50000, "batch_size": 4096,
              "target_freq": 1}
RESULTS, MODELS_DIR = "results_4x4", "models_4x4"  # these two are placeholders



def calc_G(rewards, gamma):
    G = 0.
    for r in reversed(rewards): 
        G = r + gamma*G
    return G


def decaying_epsilon_greedy(current_episode, num_episodes,e_max,e_min):
    r = max(0, (num_episodes - current_episode)/num_episodes)
    eps = e_min + (e_max - e_min) * r
    return eps


class RandomAgent:
    def __init__(self):
        return

    def select_action(self, state):
        legal = [c for c in range(state.shape[1]) if state[0, c] == 0]
        if not legal:
            raise RuntimeError("No legal moves left")
        return random.choice(legal)


def save_experiment_config(models_dir: str,
                           results_dir: str,
                           board_size: Tuple[int,int],
                           connect: int,
                           episodes_q: int,
                           agents_q: int,
                           episodes_d: int,
                           agents_d: int,
                           eps_max: float,
                           eps_min: float,
                           decay: float,
                           q_params: Dict,
                           dqn_params: Dict,
                           gamma: float = None,
                           target_freq: int = None):
    """
    Save all hyperparameters and paths into models_dir/config.json
    so each (γ, target_freq) combo has its own config.
    """
    os.makedirs(models_dir, exist_ok=True)

    config = {
        "BOARD_SIZE": board_size,
        "CONNECT": connect,
        "EPISODES_Q": episodes_q,
        "AGENTS_Q": agents_q,
        "EPISODES_DQN": episodes_d,
        "AGENTS_DQN": agents_d,
        "EPS_MAX": eps_max,
        "EPS_MIN": eps_min,
        "DECAY": decay,
        "Q_PARAMS": q_params,
        "DQN_PARAMS": dqn_params,
        "RESULTS_DIR": results_dir,
        "MODELS_DIR": models_dir,
    }
    if gamma is not None:
        config["GAMMA"] = gamma
    if target_freq is not None:
        config["TARGET_FREQ"] = target_freq

    with open(os.path.join(models_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=3)



def play_q(agent, env):
    env.reset()
    s_idx = agent._state_to_index(env.get_state())
    R, done, steps = [], False, 0
    while not done:
        a = agent.e_greedy(s_idx)
        try: 
            s_next, r, done = env.execute_action(a)
        except ValueError:
            continue
        n_idx = agent._state_to_index(s_next)
        agent.update(s_idx, a, r, n_idx)
        R.append(r)
        s_idx = n_idx
        steps += 1
    return calc_G(R, agent.gamma), (R[-1] if R else 0), steps

def play_dqn(agent, env):
    env.reset()
    s = env.get_state()
    R, done, steps = [], False, 0
    while not done:
        while True:
            a = agent.select_action(s)
            try: 
                s2, r, done = env.execute_action(a)
                break
            except ValueError: 
                continue
        agent.buffer.push(s,a,r,s2,done)
        agent._optimize()
        R.append(r)
        s = s2 
        steps += 1
    return calc_G(R, agent.gamma), (R[-1] if R else 0), steps


def collect(factory:Callable[[],object], 
            episodes:int, 
            n:int,
            eps_max, 
            eps_min, 
            board_size,
            connect)->Tuple[Dict[str,pd.DataFrame],List[object]]:
    mats = {k: np.zeros((episodes, n), float) for k in ("return","win","draw","len")}
    agents=[]
    for j in tqdm(range(n), desc="agents"):
        a = factory(); env=ConnectN(size=board_size,connect=connect)
        a.epsilon=eps_max
        agents.append(a)
        bar = tqdm(range(episodes), desc=f"agent{j+1}", leave=False)
        for ep in bar:
            if ep > 5000:
                a.epsilon = decaying_epsilon_greedy(ep, episodes,eps_max,eps_min) #max(eps_min, a.epsilon*decay)
            G, last_r, L = (play_q(a,env) if isinstance(a,QAgent)
                            else play_dqn(a,env))
            mats["return"][ep,j]=G
            mats["win"   ][ep,j]=1.0 if last_r>0 else 0.0
            mats["draw"  ][ep,j]=1.0 if last_r==0 else 0.0
            mats["len"   ][ep,j]=L

            if isinstance(a, DQNAgent):
                bar.set_postfix(ε=f"{a.epsilon:.3f}", G=f"{G:.1f}")
        bar.close()
    # convert to DataFrames
    dfs={k:pd.DataFrame(mats[k],index=[f"ep{e+1}" for e in range(episodes)],
                        columns=[f"agent{j+1}" for j in range(n)]) for k in mats}
    return dfs, agents


def save_all(dfs: Dict[str, pd.DataFrame],
             label: str,
             results_dir: str,
             save_interval: int = None):
    d = os.path.join(results_dir, label)
    os.makedirs(d, exist_ok=True)

    # 1) full-run metrics + last-50 summary
    for k, df in dfs.items():
        df.to_csv(os.path.join(d, f"metrics_{k}.csv"))
    ret  = dfs["return"].tail(50)
    mean = ret.mean(axis=0).mean()
    sem  = ret.sem(axis=0).mean()
    with open(os.path.join(d, "last50_summary.txt"), "w") as f:
        f.write(json.dumps({"mean_return": float(mean), "mean_sem": float(sem)}, indent=2))

    # 2) full-run curves
    y_labels = {"return": "G", "win": "Win", "draw": "Draw", "len": "Moves"}
    curves_dir = os.path.join(d, "curves")
    os.makedirs(curves_dir, exist_ok=True)
    for k, df in dfs.items():
        m, s = df.mean(axis=1), df.sem(axis=1)
        plt.figure()
        plt.title(f"{label} – {k}")
        plt.xlabel("Episode")
        plt.ylabel(y_labels[k])
        x = np.arange(len(m))
        plt.scatter(x, m, s=5)
        plt.fill_between(x, m - s, m + s, alpha=.3)
        plt.tight_layout()
        plt.savefig(os.path.join(curves_dir, f"{k}.png"))
        plt.close()

    # 3) interval snapshots
    if save_interval:
        interval_dir = os.path.join(d, f"every_{save_interval}")
        os.makedirs(interval_dir, exist_ok=True)
        for ep in range(save_interval, dfs["return"].shape[0] + 1, save_interval):
            ep_label = f"up_to_ep_{ep}"
            subdir = os.path.join(interval_dir, ep_label)
            os.makedirs(subdir, exist_ok=True)
            # save CSVs
            for k, df in dfs.items():
                df.iloc[:ep].to_csv(os.path.join(subdir, f"metrics_{k}.csv"))
            # save curves
            for k, df in dfs.items():
                m, s = df.mean(axis=1)[:ep], df.sem(axis=1)[:ep]
                plt.figure()
                plt.title(f"{label} – {k} (up to ep {ep})")
                plt.xlabel("Episode")
                plt.ylabel(y_labels[k])
                x = np.arange(len(m))
                plt.scatter(x, m, s=5)
                plt.fill_between(x, m - s, m + s, alpha=.3)
                plt.tight_layout()
                plt.savefig(os.path.join(subdir, f"{k}.png"))
                plt.close()



def showdown(best_q: QAgent|None, best_d: DQNAgent|None, random_a: RandomAgent|None, games: int = 1000, board_size: Tuple[int,int] = (4, 4), connect: int = 4) -> Tuple[int,int,int]:
    """
    Plays `games` matches between best_q and best_d.
    Alternates the starting player each game.
    Returns (q_wins, d_wins, draws).
    """
    q_wins = d_wins = r_wins = draws = 0

    if best_q is None and (best_d is not None and random_a is not None):
        mode = "dqn vs random"
    elif best_d is None and (best_q is not None and random_a is not None):
        mode = "q vs random"
    elif best_q is not None and best_d is not None:
        mode = "q vs dqn"

    if mode == "q vs dqn":
        for g in tqdm(range(games), desc=f"showdown {mode}"):
            env = ConnectN(size=board_size, connect=connect)
            env.reset()
            state = env.get_state()
            # alternate who starts
            current = "Q" if (g % 2 == 0) else "D"
            starter = "Q" if (g % 2 == 0) else "D"
            while True:
                # pick an action
                if current == "Q":
                    a   = best_q.greedy_action(state)
                else:
                    # DQNAgent picks the greedy network action
                    a   = best_d.predict_action(state)
                state, _, done = env.execute_action(a)
                if done:
                    outcome = env.is_game_over()
                    if outcome == 1:
                        # player 1’s piece — starter wins
                        if starter == "Q":
                            q_wins += 1
                        else:
                            d_wins += 1
                    elif outcome == 2:
                        # player 2’s piece — the other wins
                        if starter == "Q":
                            d_wins += 1
                        else:
                            q_wins += 1
                    else:
                        draws += 1
                    break
                # switch turns
                current = "D" if current == "Q" else "Q"
        return q_wins, d_wins, draws
    
    if mode == "q vs random":
        for g in tqdm(range(games), desc=f"showdown {mode}"):
            env = ConnectN(size=board_size, connect=connect)
            env.reset()
            state = env.get_state()
            # alternate who starts 
            current = "Q" if (g % 2 == 0) else "R"
            starter = "Q" if (g % 2 == 0) else "R"
            while True:
                # pick an action
                if current == "Q":
                    a   = best_q.greedy_action(state)
                    q_state = state.copy()
                else:
                    # DQNAgent picks the greedy network action
                    a   = random_a.select_action(state)
                # execute it
                try: state, _, done = env.execute_action(a)
                except: raise ValueError(f"current: {current}, action: {a}, q state: {q_state}, state: {state}")
                if done:
                    outcome = env.is_game_over()
                    if outcome == 1:
                        # player 1’s piece — starter wins
                        if starter == "Q":
                            q_wins += 1
                        else:
                            r_wins += 1
                    elif outcome == 2:
                        # player 2’s piece — the other wins
                        if starter == "Q":
                            r_wins += 1
                        else:
                            q_wins += 1
                    else:
                        draws += 1
                    break
                # switch turns
                current = "R" if current == "Q" else "Q"
        return q_wins, r_wins, draws
      
    if mode == "dqn vs random":
        for g in tqdm(range(games), desc=f"showdown {mode}"):
            env = ConnectN(size=board_size, connect=connect)
            env.reset()
            state = env.get_state()
            # alternate who starts 
            current = "D" if (g % 2 == 0) else "R"
            starter = "D" if (g % 2 == 0) else "R"
            while True:
                # pick an action
                if current == "D":
                    # QAgent uses its epsilon‐greedy policy with epsilon=0
                    a   = best_d.predict_action(state)
                    d_state = state.copy()
                else:
                    # DQNAgent picks the greedy network action
                    a   = random_a.select_action(state)
                #state, _, done = env.execute_action(a)
                try: state, _, done = env.execute_action(a)
                except: raise ValueError(f"current: {current}, action: {a}, d state: {d_state}, state: {state}")
                if done:
                    outcome = env.is_game_over()
                    if outcome == 1:
                        # player 1’s piece — starter wins
                        if starter == "D":
                            d_wins += 1
                        else:
                            r_wins += 1
                    elif outcome == 2:
                        # player 2’s piece — the other wins
                        if starter == "D":
                            r_wins += 1
                        else:
                            d_wins += 1
                    else:
                        draws += 1
                    break
                # switch turns
                current = "R" if current == "D" else "D"
        return d_wins, r_wins, draws

def plot_showdown(stats,folder,label):
    lab=["Q wins","DQN wins","Draws"]
    plt.figure()
    plt.title(f"Q vs DQN outcome ({label})")
    plt.ylabel("count")
    plt.bar(lab,stats,color=["red","blue","gray"])
    plt.tight_layout()
    plt.savefig(os.path.join(folder,"q_vs_dqn_outcome.png"))
    plt.close()

def plot_showdown_random_q(stats,folder,label):
    lab=["Q wins","Random wins","Draws"]
    plt.figure()
    plt.title(f"Q vs Random outcome ({label})")
    plt.ylabel("count")
    plt.bar(lab,stats,color=["red","blue","gray"])
    plt.tight_layout()
    plt.savefig(os.path.join(folder,"q_vs_random_outcome.png"))
    plt.close()

def plot_showdown_random_dqn(stats,folder, label):
    lab=["DQN wins","Random wins","Draws"]
    plt.figure()
    plt.title(f"DQN vs Random outcome ({label})")
    plt.ylabel("count")
    plt.bar(lab,stats,color=["red","blue","gray"])
    plt.tight_layout()
    plt.savefig(os.path.join(folder,"dqn_vs_random_outcome.png"))
    plt.close()
    

def run_q(episodes: int,
          n_agents: int,
          eps_max: float,
          eps_min: float,
          decay: float,
          q_params: Dict,
          board_size: Tuple[int,int],
          connect: int,
          results_dir: str,
          models_dir: str,
          label: str,
          save_interval: int = None) -> QAgent:
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir,  exist_ok=True)

    fac = lambda: QAgent(ConnectN(board_size, connect), **q_params)
    dfs, agents = collect(fac, episodes, n_agents, eps_max, eps_min, board_size,connect)

    save_all(dfs,
             label=f"QAgent_{label}",
             results_dir=results_dir,
             save_interval=save_interval)

    np.save(os.path.join(models_dir, "replica_Q_best.npy"),
            agents[0].q_table)
    return agents[0]


def run_d(episodes: int,
          n_agents: int,
          eps_max: float,
          eps_min: float,
          decay: float,
          dqn_params: Dict,
          board_size: Tuple[int,int],
          connect: int,
          results_dir: str,
          models_dir: str,
          label: str,
          save_interval: int = None,
          run_mode: str = 'all') -> DQNAgent:
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir,  exist_ok=True)
    if run_mode == 'all':
        fac = lambda: DQNAgent(ConnectN(board_size, connect), **dqn_params)
        dfs, agents = collect(fac, episodes, n_agents, eps_max, eps_min, board_size,connect)

        save_all(dfs,
                label=f"DQNAgent_{label}",
                results_dir=results_dir,
                save_interval=save_interval)

        torch.save(agents[0].policy_net.state_dict(),
                os.path.join(models_dir, "replica_DQN_best.pt"))
        return agents[0]
    if run_mode == 'save_dqn_mva':
        dqn_res_dir = os.path.join(results_dir, f"DQNAgent_{label}")
        print(f'res_dir: {dqn_res_dir}')
        os.makedirs(dqn_res_dir, exist_ok=True)
        curves_dir = os.path.join(dqn_res_dir, "curves")
        print(f'curves_dir: {curves_dir}')
        os.makedirs(curves_dir, exist_ok=True)
        return_df = pd.read_csv(os.path.join(dqn_res_dir, "metrics_return.csv"))
        len_df = pd.read_csv(os.path.join(dqn_res_dir, "metrics_len.csv"))
        return_df = return_df.iloc[:, 1]
        expanding_return = return_df.expanding(min_periods=1).mean()
        return_df = return_df.rolling(window=50).mean().combine_first(expanding_return)
        m = return_df
        #save return mva
        plt.figure()
        plt.title(f"{label} - return (moving average)")
        plt.xlabel("Episode")
        plt.ylabel("G")
        x = np.arange(len(m))
        plt.ylim(0,50)
        plt.plot(x, m)
        plt.tight_layout()
        plt.savefig(os.path.join(curves_dir, f"return_mva.png"))
        plt.close()
        #save len mva
        len_df = len_df.iloc[:, 1]
        expanding_len = len_df.expanding(min_periods=1).mean()
        len_df = len_df.rolling(window=50).mean().combine_first(expanding_len)
        m = len_df
        plt.figure()
        plt.title(f"{label} - len (moving average)")
        plt.xlabel("Episode")
        plt.ylabel("Moves")
        x = np.arange(len(m))
        plt.ylim(0,50)
        plt.plot(x, m)
        plt.tight_layout()
        plt.savefig(os.path.join(curves_dir, f"len_mva.png"))
        plt.close()




def sweep_experiments(gammas: List[float],
                      target_freqs: List[int],
                      episodes_q: int,
                      agents_q: int,
                      episodes_d: int,
                      agents_d: int,
                      eps_max: float,
                      eps_min: float,
                      decay: float,
                      q_params: Dict,
                      dqn_params: Dict,
                      board_size: Tuple[int,int],
                      connect: int,
                      results_root: str,
                      models_root:  str,
                      save_interval: int,
                      showdown_games: int,
                      mode: str) -> None:
    """
    Run run_q/run_d over all combinations of gammas and target_freqs.
    Each combination gets its own subfolders under results_root and models_root.
    """
    
    for gamma in gammas:
        for tf in target_freqs:
            # --- prepare hyperparams & dirs ---
            qp = q_params.copy(); qp["gamma"] = gamma
            dp = dqn_params.copy()
            dp["gamma"]       = gamma
            dp["target_freq"] = tf

            label   = f"g{gamma}_tf{tf}"
            res_dir = os.path.join(results_root, label)
            mdl_dir = os.path.join(models_root,  label)

            if mode == "full":
                save_experiment_config(
                    models_dir = mdl_dir,
                    results_dir = res_dir,
                    board_size = board_size,
                    connect = connect,
                    episodes_q = episodes_q,
                    agents_q = agents_q,
                    episodes_d = episodes_d,
                    agents_d = agents_d,
                    eps_max = eps_max,
                    eps_min = eps_min,
                    decay = decay,
                    q_params = qp,
                    dqn_params = dp,
                    gamma = gamma,
                    target_freq = tf
                )

                print(f"\n=== Experiment γ={gamma}, target_freq={tf} ===")
                # 1) train Q
                q_agent = run_q( episodes=episodes_q,
                                n_agents=agents_q,
                                eps_max=eps_max,
                                eps_min=eps_min,
                                decay=decay,
                                q_params=qp,
                                board_size=board_size,
                                connect=connect,
                                results_dir=res_dir,
                                models_dir=mdl_dir,
                                label=label,
                                save_interval=save_interval)

                # 2) train DQN
                d_agent = run_d( episodes=episodes_d,
                                n_agents=agents_d,
                                eps_max=eps_max,
                                eps_min=eps_min,
                                decay=decay,
                                dqn_params=dp,
                                board_size=board_size,
                                connect=connect,
                                results_dir=res_dir,
                                models_dir=mdl_dir,
                                label=label,
                                save_interval=save_interval)
                
                random_agent = RandomAgent()

                # 3) showdown
                print(f"--- Showdown ({showdown_games} games) for {label} ---")
                q_wins, d_wins, draws = showdown(q_agent, d_agent, None, showdown_games,board_size,connect)

                # 4) save showdown results
                out = os.path.join(res_dir,label,"showdown")
                os.makedirs(out, exist_ok=True)
                with open(os.path.join(out, "counts.json"), "w") as f:
                    json.dump({"Q wins": q_wins,
                            "DQN wins": d_wins,
                            "draws":    draws}, f, indent=2)
                plot_showdown((q_wins, d_wins, draws), out, label)

                # random agent vs Q agent
                q_wins, r_wins, draws = showdown(q_agent, None, random_agent, showdown_games,board_size,connect)
                out = os.path.join(res_dir,label,"showdown_random_q")
                os.makedirs(out, exist_ok=True)
                with open(os.path.join(out, "counts.json"), "w") as f:
                    json.dump({"Q wins": q_wins,
                            "Random wins": r_wins,
                            "draws": draws}, f, indent=2)
                plot_showdown_random_q((q_wins, r_wins, draws), out, label)

                # random agent vs DQN agent
                d_wins, r_wins, draws = showdown(None, d_agent, random_agent, showdown_games,board_size,connect)
                out = os.path.join(res_dir,label,"showdown_random_dqn")
                os.makedirs(out, exist_ok=True)
                with open(os.path.join(out, "counts.json"), "w") as f:
                    json.dump({"DQN wins": d_wins,
                            "Random wins": r_wins,
                            "draws": draws}, f, indent=2)
                plot_showdown_random_dqn((d_wins, r_wins, draws), out, label)
    
            if mode=="showdown":
                # load the best Q and DQN agents from the last experiment
                q_agent, d_agent = load_trained_agents(mdl_dir)
                random_agent = RandomAgent()

                # showdown between Q and DQN agents
                q_wins, d_wins, draws = showdown(q_agent, d_agent, None, showdown_games,board_size,connect)
                out = os.path.join(results_root,label,"showdown")
                print(out)
                os.makedirs(out, exist_ok=True)
                with open(os.path.join(out, "counts.json"), "w") as f:
                    json.dump({"Q wins": q_wins,
                            "DQN wins": d_wins,
                            "draws":    draws}, f, indent=2)
                plot_showdown((q_wins, d_wins, draws),out,label)

                # random agent vs Q agent
                q_wins, r_wins, draws = showdown(q_agent, None, random_agent, showdown_games,board_size,connect)
                out = os.path.join(results_root,label,"showdown_random_q")
                print(out)
                os.makedirs(out, exist_ok=True)
                with open(os.path.join(out, "counts.json"), "w") as f:
                    json.dump({"Q wins": q_wins,
                            "Random wins": r_wins,
                            "draws": draws}, f, indent=2)
                plot_showdown_random_q((q_wins, r_wins, draws),out,label)

                # random agent vs DQN agent
                d_wins, r_wins, draws = showdown(None, d_agent, random_agent, showdown_games,board_size,connect)
                out = os.path.join(results_root,label,"showdown_random_dqn")
                print(out)
                os.makedirs(out, exist_ok=True)
                with open(os.path.join(out, "counts.json"), "w") as f:
                    json.dump({"DQN wins": d_wins,
                            "Random wins": r_wins,
                            "draws": draws}, f, indent=2)
                plot_showdown_random_dqn((d_wins, r_wins, draws),out,label)

            if mode == "save_dqn_mva":
                d_agent = run_d( episodes=episodes_d,
                                n_agents=agents_d,
                                eps_max=eps_max,
                                eps_min=eps_min,
                                decay=decay,
                                dqn_params=dp,
                                board_size=board_size,
                                connect=connect,
                                results_dir=res_dir,
                                models_dir=mdl_dir,
                                label=label,
                                save_interval=save_interval,
                                run_mode='save_dqn_mva')
            


def load_trained_agents(models_dir: str):
    """
    Load a QAgent and a DQNAgent from saved files in models_dir.
    Expects:
      - replica_Q_best.npy   (the Q-table)
      - replica_DQN_best.pt  (the policy_net state_dict)
    Returns: (q_agent, dqn_agent)
    """
    # Q-Agent
    q_agent = QAgent(ConnectN(BOARD_SIZE, CONNECT), **Q_PARAMS)
    q_path   = os.path.join(models_dir, "replica_Q_best.npy")
    q_agent.q_table = np.load(q_path)

    # DQN-Agent
    d_agent = DQNAgent(ConnectN(BOARD_SIZE, CONNECT), **DQN_PARAMS)
    d_path  = os.path.join(models_dir, "replica_DQN_best.pt")
    sd      = torch.load(d_path, map_location=d_agent.device)
    d_agent.policy_net.load_state_dict(sd)
    # sync target_net & set eval mode
    d_agent.target_net.load_state_dict(d_agent.policy_net.state_dict())
    d_agent.policy_net.eval()
    d_agent.target_net.eval()

    # No exploration at showdown
    q_agent.epsilon = 0.0
    d_agent.epsilon = 0.0

    return q_agent, d_agent




# main ----------------------------------------------------------------------
if __name__=="__main__":
    start=time.time()
    print("CUDA:",torch.cuda.is_available())
    BOARD_SIZES = [(4, 4),(5, 5),(6, 7)] #
    CONNECTS = [4, 3, 4] #4,3,
    GAMMAS       = [0.99]
    TARGET_FREQS = [1, 5, 10]
    SAVE_INT     = 10000

    for (board_size, connect) in zip(BOARD_SIZES, CONNECTS):
        results_dir, models_dir = f"results_{board_size[0]}x{board_size[1]}", f"models_{board_size[0]}x{board_size[1]}"
        print(f"\n=== Experiment {board_size[0]}x{board_size[1]} ===")
        sweep_experiments(
            gammas        = GAMMAS,
            target_freqs  = TARGET_FREQS,
            episodes_q    = EPISODES_Q,
            agents_q      = AGENTS_Q,
            episodes_d    = EPISODES_DQN,
            agents_d      = AGENTS_DQN,
            eps_max       = EPS_MAX,
            eps_min       = EPS_MIN,
            decay         = DECAY,
            q_params      = Q_PARAMS,
            dqn_params    = DQN_PARAMS,
            board_size    = board_size,
            connect       = connect,
            results_root  = results_dir,
            models_root   = models_dir,
            save_interval = SAVE_INT,
            showdown_games = 1000,
            mode="full" # "full", "showdown", "save_dqn_mva"
        )
