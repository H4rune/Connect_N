import numpy as np, torch, torch.nn as nn, torch.optim as optim, random
# from torch.cuda.amp import GradScaler, autocast
from torch.amp import GradScaler, autocast
import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque
import logging 

import torch._dynamo
torch._dynamo.config.suppress_errors = True


class ReplayBuffer:
    """
    A ring buffer that allocates its numpy arrays based on the first pushed state.
    Subsequent pushes must have the same state/next_state shape.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.pos      = 0
        self.size     = 0
        self._inited  = False

    def _init_buffers(self, state):
        # convert to numpy to get shape
        st = np.array(state, copy=False)
        shape = st.shape
        # allocate arrays
        self.states      = np.zeros((self.capacity, *shape), dtype=st.dtype)
        self.next_states = np.zeros_like(self.states)
        self.actions     = np.zeros(self.capacity, dtype=np.int64)
        self.rewards     = np.zeros(self.capacity, dtype=np.float32)
        self.dones       = np.zeros(self.capacity, dtype=bool)
        self._inited     = True

    def push(self, state, action, reward, next_state, done):
        if not self._inited:
            self._init_buffers(state)

        st  = np.array(state,      copy=False)
        st2 = np.array(next_state, copy=False)

        self.states[self.pos]      = st
        self.next_states[self.pos] = st2
        self.actions[self.pos]     = action
        self.rewards[self.pos]     = reward
        self.dones[self.pos]       = done

        self.pos  = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs],
            self.dones[idxs],
        )

    def __len__(self):
        return self.size


class DQN(nn.Module):
    """Simple CNN → FC approximator for Connect‑N."""
    def __init__(self, input_shape, num_actions,*args, **kwargs):
        super().__init__()
        c, h, w = input_shape
        self._in_channels = c                    

        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        conv_out = self._conv_features(h, w)
        self.fc = nn.Sequential(
            nn.Linear(conv_out, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

        self.apply(self._init_weights)

    
    def _conv_features(self, h, w):
        """
        Pass a dummy (batch=1, channels=c, H, W) tensor through conv stack
        to compute flattened feature size.
        """
        with torch.no_grad():
            dummy = torch.zeros(1, self._in_channels, h, w)
            out = self.conv(dummy)
            return int(out.view(1, -1).size(1))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    
    def forward(self, x):
        x = self.conv(x)
        #x = x.view(x.size(0), -1)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)



class DQNAgent:
    """Deep Q‑learning agent with target network & replay memory."""
    def __init__(self, 
                 env,
                 epsilon=1.0, 
                 epsilon_decay=0.995, 
                 epsilon_min=0.1,
                 lr=1e-3, 
                 gamma=0.99,
                 buffer_cap=10000, 
                 batch_size=2048,
                 target_freq=5,
                 tau = 0.005, 
                 device=None):

        
        self.env            = env
        self.epsilon        = epsilon
        self.decay          = epsilon_decay
        self.epsilon_min    = epsilon_min
        self.gamma          = gamma
        self.batch_size     = batch_size
        self.target_freq    = target_freq
        self.tau            = tau
        self.device         = (device if device is not None
                               else torch.device('cuda' if torch.cuda.is_available()
                                                 else 'cpu'))
        if self.device.type == "cuda":
            #torch.cuda.init()                    # ensure early init
            print("DQNAgent using GPU:", torch.cuda.get_device_name(0))
        else:
            print("DQNAgent using CPU")

        
        h, w = env.height, env.width
        self.n_actions  = w
        self.policy_net = DQN((2, h, w), self.n_actions).to(self.device,memory_format=torch.channels_last)
        self.target_net = DQN((2, h, w), self.n_actions).to(self.device,memory_format=torch.channels_last)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net = torch.compile(self.policy_net,backend="eager")
        self.target_net = torch.compile(self.target_net,backend="eager")
        self.target_net.eval()

        self.opt = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_cap)
        self.episodes = 0  # counter for target updates
        self.loss_func = nn.HuberLoss()
        self.scaler = GradScaler()  # for mixed precision training


    def _state_tensor(self, board):
        """
        Return tensor on self.device with shape (1,2,H,W).
        Ensures dtype is float32 so Conv2d's weights/bias match.
        """
        b =  torch.from_numpy(board) #.to(torch.float32)
        # build two float32 channels
        channel1 = (b == 1).to(torch.float32)
        channel2 = (b == 2).to(torch.float32)
        channels   = torch.stack([channel1, channel2], dim=0)
        t        = channels.unsqueeze(0)  # shape (1,2,H,W), dtype float32    torch.from_numpy(channels)
        return t.to(self.device, non_blocking=True) #.contiguous(memory_format=torch.channels_last)

    def _valid_cols(self):
        return [c for c in range(self.n_actions)
                if self.env.board_state[0, c] == 0]

    
    def select_action(self, state):
        valids = self._valid_cols()
        if not valids:              
            return 0

        if random.random() < self.epsilon:
            return random.choice(valids)

        with torch.no_grad():
            q = self.policy_net(self._state_tensor(state)).cpu().squeeze()
            q_invalid_mask = torch.tensor(
                [(-np.inf) if c not in valids else 0.0
                 for c in range(self.n_actions)])
            q = q + q_invalid_mask
            return int(torch.argmax(q).item())
    
    def _update_target(self): # soft update of target network with moving average
        with torch.no_grad():
            for tgt_p, src_p in zip(self.target_net.parameters(),
                                    self.policy_net.parameters()):
                tgt_p.data.mul_(1 - self.tau)
                tgt_p.data.add_(self.tau * src_p.data)

    def _optimize(self): #basically equivalent to the Q-learning update
        if len(self.buffer) < self.batch_size:
            return
        batch = self.buffer.sample(self.batch_size)
        s, a, r, s2, d = batch #zip(*batch)

        #s   = torch.cat([self._state_tensor(x) for x in s], dim=0)
        #s2  = torch.cat([self._state_tensor(x) for x in s2], dim=0)


        s  = torch.from_numpy(np.stack([ (s == 1), (s == 2) ], axis=1).astype(np.float32)).to(self.device, non_blocking=True)
        s2 = torch.from_numpy(np.stack([ (s2 == 1), (s2 == 2) ], axis=1).astype(np.float32)).to(self.device, non_blocking=True)
        
        # s  = torch.from_numpy(np.stack(s, axis=0)).to(self.device, non_blocking=True)
        # s2 = torch.from_numpy(np.stack(s2, axis=0)).to(self.device, non_blocking=True)
        a   = torch.tensor(a, device=self.device).unsqueeze(1)
        r   = torch.tensor(r, device=self.device).unsqueeze(1)
        d   = torch.tensor(d, device=self.device).unsqueeze(1).float()

        self.opt.zero_grad(set_to_none=True) #set_to_none=True
        with autocast(device_type=self.device.type, dtype=torch.bfloat16):
            q_pred = self.policy_net(s).gather(1, a)
            with torch.no_grad():
                q_next = self.target_net(s2).max(1, keepdim=True)[0]
                q_targ = r + self.gamma * q_next * (1 - d)
            loss = self.loss_func(q_pred, q_targ) #MSELoss()

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        # self.scaler.step(self.opt)
        # loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        # self.opt.step()
        self.scaler.step(self.opt)
        self.scaler.update()

    def generate_episode(self, max_steps=1000, training=True, render=False):
        self.env.reset()
        state = self.env.get_state()
        done  = False
        step  = 0
        total = 0

        while not done and step < max_steps:
            while True:                                   
                act = self.select_action(state)
                try:
                    nxt, rew, done = self.env.execute_action(act)
                    break
                except ValueError:
                    continue

            if training:
                #state  = self.buffer.board_to_channels(state)
                #nxt  = self.buffer.board_to_channels(nxt)
                self.buffer.push(state, act, rew, nxt, done)
                self._optimize()

            state = nxt
            total += rew
            step += 1
            if render:
                self.env.display_board(pretty=True)

        # episode end bookkeeping
        if training:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.decay)
            self.episodes += 1
            if self.episodes % self.target_freq == 0:
                # self.target_net.load_state_dict(self.policy_net.state_dict())
                self._update_target()
        return total
    

    def predict_action(self, state):
        """
        Predict the greedy action for a given board state, masking out full columns.
        """
        with torch.no_grad():
            q = self.policy_net(self._state_tensor(state)).cpu().squeeze()                          
        legal = [c for c in range(self.n_actions) if state[0, c] == 0]
        mask = torch.full_like(q, -float("inf"))
        mask[legal] = 0.0
        q = q + mask
        return int(torch.argmax(q).item())
