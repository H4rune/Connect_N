
import os, time, json
from typing import List, Dict, Tuple, Callable

import numpy as np
import pandas as pd                                 # NEW
import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm

from ConnectN   import ConnectN
from Agent_Q    import QAgent
from Agent_DQN  import DQNAgent


# ────────────────── config (edit freely) ──────────────────
BOARD_SIZE, CONNECT  = (6, 7), 4
EPISODES_Q,  AGENTS_Q  = 10000, 50
EPISODES_DQN, AGENTS_DQN = 10000, 5
EPS_MAX, EPS_MIN, DECAY = 1.0, 0.05, 0.995
Q_PARAMS   = {"epsilon": EPS_MAX, "alpha": 0.2, "gamma": 0.95}
DQN_PARAMS = {"epsilon": EPS_MAX, "epsilon_decay": DECAY,
              "epsilon_min": EPS_MIN, "lr": 1e-3, "gamma": 0.99,
              "buffer_cap": 20000, "batch_size": 128,
              "target_freq": 10}
RESULTS, MODELS_DIR = "results", "models"


# ──────────────────────────────────────────────────────────
def calc_G(rewards, gamma):
    G = 0.
    for r in reversed(rewards): 
        G = r + gamma*G
    return G


def decaying_epsilon_greedy(current_episode, num_episodes,e_max,e_min):
    r = max(0, (num_episodes - current_episode)/num_episodes)
    eps = e_min + (e_max - e_min) * r
    return eps

# episode runners ------------------------------------------------
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
        R.append(r); s_idx = n_idx
        steps += 1
    return calc_G(R, agent.gamma), (R[-1] if R else 0), steps

def play_dqn(agent, env):
    env.reset(); s = env.get_state()
    R, done, steps = [], False, 0
    while not done:
        while True:
            a = agent.select_action(s)
            try: s2, r, done = env.execute_action(a); break
            except ValueError: continue
        agent.buffer.push(s,a,r,s2,done)
        agent._optimize()
        R.append(r); s = s2
        steps += 1
    return calc_G(R, agent.gamma), (R[-1] if R else 0), steps


# metric collection ---------------------------------------------------------
def collect(factory:Callable[[],object], 
            episodes:int, 
            n:int,
            eps_max, 
            eps_min, 
            decay)->Tuple[Dict[str,pd.DataFrame],List[object]]:
    mats = {k: np.zeros((episodes, n), float) for k in ("return","win","draw","len")}
    agents=[]
    for j in tqdm(range(n), desc="agents"):
        a = factory(); env=ConnectN(size=BOARD_SIZE,connect=CONNECT)
        a.epsilon=eps_max
        agents.append(a)
        for ep in range(episodes):
            a.epsilon = decaying_epsilon_greedy(ep, episodes,eps_max,eps_min) #max(eps_min, a.epsilon*decay)
            G, last_r, L = (play_q(a,env) if isinstance(a,QAgent)
                            else play_dqn(a,env))
            mats["return"][ep,j]=G
            mats["win"   ][ep,j]=1.0 if last_r==1 else 0.0
            mats["draw"  ][ep,j]=1.0 if last_r==0 else 0.0
            mats["len"   ][ep,j]=L
    # convert to DataFrames
    dfs={k:pd.DataFrame(mats[k],index=[f"ep{e+1}" for e in range(episodes)],
                        columns=[f"agent{j+1}" for j in range(n)]) for k in mats}
    return dfs, agents


# save + analyse ------------------------------------------------------------
def save_all(dfs:Dict[str,pd.DataFrame], label:str):
    d= os.path.join(RESULTS,label)
    os.makedirs(d,exist_ok=True)
    for k,df in dfs.items():
        df.to_csv(os.path.join(d,f"metrics_{k}.csv"))
    # last-50 stats for 'return'
    ret=dfs["return"].iloc[-50:]
    mean=ret.mean(axis=0).mean()
    sem=ret.sem(axis=0).mean()
    with open(os.path.join(d,"last50_summary.txt"),"w") as f:
        f.write(json.dumps({"mean_return":float(mean),"mean_sem":float(sem)},indent=2))
    # curves
    curves=os.path.join(d,"curves")
    os.makedirs(curves,exist_ok=True)
    for k,df in dfs.items():
        m, s = df.mean(axis=1), df.sem(axis=1)
        plt.figure()
        plt.title(f"{label} – {k}")
        plt.xlabel("Episode")
        plt.ylabel({"return":"G","win":"Win","draw":"Draw","len":"Moves"}[k])
        plt.plot(m)
        plt.fill_between(m.index, m-s, m+s, alpha=.3)
        plt.tight_layout()
        plt.savefig(os.path.join(curves,f"{k}.png")); plt.close()


# cross-play showdown -------------------------------------------------------
def showdown(best_q:QAgent, best_d:DQNAgent, games=1000):
    q_win=d_win=draw=0
    for g in tqdm(range(games),desc="showdown"):
        env=ConnectN(size=BOARD_SIZE,connect=CONNECT)
        env.reset()
        state=env.get_state()
        done=False
        current="Q" if g%2==0 else "D"
        while not done:
            if current=="Q":
                row=best_q.q_table[best_q._state_to_index(state)]
                legal=[c for c in range(env.width) if env.board_state[0,c]==0]
                row[[c for c in range(env.width) if c not in legal]]=-np.inf
                a=int(np.argmax(row))
            else:
                legal=[c for c in range(env.width) if env.board_state[0,c]==0]
                with torch.no_grad():
                    q=best_d.policy_net(best_d._state_tensor(state)).cpu().squeeze()
                q[[c for c in range(env.width) if c not in legal]]=-np.inf
                a=int(torch.argmax(q))
            try: 
                state,r,done=env.execute_action(a)
            except ValueError: 
                continue
            current="D" if current=="Q" else "Q"
        if r==1: 
            (q_win,d_win)[current=="Q"] +=1
        elif r==0: 
            draw+=1
    return q_win,d_win,draw

def plot_showdown(stats,folder):
    lab=["Q wins","DQN wins","Draws"]
    plt.figure()
    plt.title("Q vs DQN outcome")
    plt.ylabel("count")
    plt.bar(lab,stats,color=["red","blue","gray"])
    plt.tight_layout()
    plt.savefig(os.path.join(folder,"q_vs_dqn_outcome.png"))
    plt.close()

# runners -------------------------------------------------------------------
def run_q():
    fac=lambda:QAgent(ConnectN(BOARD_SIZE,CONNECT),**Q_PARAMS)
    dfs,agents=collect(fac,EPISODES_Q,AGENTS_Q,EPS_MAX,EPS_MIN,DECAY)
    save_all(dfs,"QAgent")
    os.makedirs(MODELS_DIR,exist_ok=True)
    np.save(os.path.join(MODELS_DIR,"replica_Q_best.npy"),agents[0].q_table)
    return agents[0]

def run_d():
    fac=lambda:DQNAgent(ConnectN(BOARD_SIZE,CONNECT),**DQN_PARAMS)
    dfs,agents=collect(fac,EPISODES_DQN,AGENTS_DQN,EPS_MAX,EPS_MIN,DECAY)
    save_all(dfs,"DQNAgent")
    os.makedirs(MODELS_DIR,exist_ok=True)
    torch.save(agents[0].policy_net.state_dict(),
               os.path.join(MODELS_DIR,"replica_DQN_best.pt"))
    return agents[0]
# main ----------------------------------------------------------------------
if __name__=="__main__":
    start=time.time(); print("CUDA:",torch.cuda.is_available())
    best_q=run_q()
    best_d=run_d()
    q,d,draw=showdown(best_q,best_d,500)
    out_dir=os.path.join(RESULTS,"showdown")
    os.makedirs(out_dir,exist_ok=True)
    with open(os.path.join(out_dir,"counts.json"),"w") as f:
        json.dump({"Q wins":q,"DQN wins":d,"draws":draw},f,indent=2)
    plot_showdown((q,d,draw),out_dir)
    print("Showdown counts:",q,d,draw)
    print("Finished in",round(time.time()-start,1),"s")
