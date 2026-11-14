from collections import deque
import random
import numpy as np
import torch
from torch import nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(),
                                 nn.Linear(128, 128), nn.ReLU(),
                                 nn.Linear(128, action_dim))
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, config, state_dim=2, device=None):
        self.cfg = config
        self.action_dim = config.max_order + 1
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.online = QNetwork(state_dim, self.action_dim).to(self.device)
        self.target = QNetwork(state_dim, self.action_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.memory = deque(maxlen=config.replay_capacity)
        self.steps = 0

    def epsilon(self):
        frac = min(1.0, self.steps / self.cfg.epsilon_decay_steps)
        return self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    def act(self, state, valid_actions, explore=False):
        valid = np.asarray(valid_actions, dtype=int)
        if explore and random.random() < self.epsilon():
            return int(np.random.choice(valid))
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.online(x).squeeze(0).cpu().numpy()
        q[[a for a in range(self.action_dim) if a not in valid]] = -1e9
        return int(np.argmax(q))

    def remember(self, state, action, reward, next_state, done, next_valid):
        self.memory.append((state, action, reward, next_state, done, next_valid))

    def learn(self):
        if len(self.memory) < self.cfg.batch_size:
            return None
        batch = random.sample(self.memory, self.cfg.batch_size)
        states, actions, rewards, next_states, dones, next_valids = zip(*batch)
        states = torch.tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.asarray(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        q_values = self.online(states).gather(1, actions).squeeze(1)
        # Invalid actions (orders above the free capacity) are masked in one
        # vectorised step instead of a Python loop over the batch.
        mask = np.ones((len(next_valids), self.action_dim), dtype=bool)
        for i, valid in enumerate(next_valids):
            mask[i, np.asarray(list(valid), dtype=int)] = False
        mask = torch.tensor(mask, device=self.device)
        with torch.no_grad():
            next_q = self.target(next_states).masked_fill(mask, -1e9)
            max_next = next_q.max(dim=1).values
            target = rewards + self.cfg.gamma * max_next * (1.0 - dones)
        loss = nn.functional.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 5.0)
        self.optimizer.step()
        self.steps += 1
        if self.steps % self.cfg.target_update_steps == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def save(self, path):
        torch.save(self.online.state_dict(), path)

    def load(self, path):
        self.online.load_state_dict(torch.load(path, map_location=self.device))
        self.target.load_state_dict(self.online.state_dict())
