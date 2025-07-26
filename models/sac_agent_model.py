import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import logging

# Setup logging
logging.basicConfig(
    filename='logs/sac_agent.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Neural Networks --------

class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64, log_std_min=-20, log_std_max=2):
        super(GaussianPolicy, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # reparameterization trick
        y_t = torch.tanh(x_t)   # bounded actions between -1 and 1
        action = y_t
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        mean = torch.tanh(mean)
        return action, log_prob, mean


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        q_value = self.net(x)
        return q_value


# -------- SAC Agent --------

class SACAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        replay_buffer_size=100000,
        batch_size=64,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,  # entropy temperature
        lr=3e-4,
        automatic_entropy_tuning=True,
        target_update_freq=1,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.target_update_freq = target_update_freq
        self.automatic_entropy_tuning = automatic_entropy_tuning

        # Replay buffer
        self.memory = deque(maxlen=replay_buffer_size)

        # Networks
        self.policy = GaussianPolicy(state_dim, action_dim).to(DEVICE)
        self.q1 = QNetwork(state_dim, action_dim).to(DEVICE)
        self.q2 = QNetwork(state_dim, action_dim).to(DEVICE)
        self.q1_target = QNetwork(state_dim, action_dim).to(DEVICE)
        self.q2_target = QNetwork(state_dim, action_dim).to(DEVICE)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)

        # Entropy tuning
        if self.automatic_entropy_tuning:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        else:
            self.alpha = alpha

        self.learn_step = 0
        logger.info(f"SAC Agent initialized with state_dim={state_dim}, action_dim={action_dim}")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        if evaluate:
            with torch.no_grad():
                mean, _ = self.policy.forward(state)
                action = torch.tanh(mean)
                return action.cpu().numpy()[0]
        else:
            action, _, _ = self.policy.sample(state)
            return action.cpu().detach().numpy()[0]

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(DEVICE)
        actions = torch.FloatTensor(np.array(actions)).to(DEVICE)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(DEVICE)
        next_states = torch.FloatTensor(np.array(next_states)).to(DEVICE)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(DEVICE)

        # --- Update Q networks ---
        with torch.no_grad():
            next_action, next_log_prob, _ = self.policy.sample(next_states)
            target_q1 = self.q1_target(next_states, next_action)
            target_q2 = self.q2_target(next_states, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_value = rewards + (1 - dones) * self.gamma * target_q

        q1_value = self.q1(states, actions)
        q2_value = self.q2(states, actions)
        q1_loss = nn.MSELoss()(q1_value, target_value)
        q2_loss = nn.MSELoss()(q2_value, target_value)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # --- Update policy network ---
        new_action, log_prob, _ = self.policy.sample(states)
        q1_new = self.q1(states, new_action)
        q2_new = self.q2(states, new_action)
        q_new = torch.min(q1_new, q2_new)

        policy_loss = ((self.alpha * log_prob) - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # --- Adjust alpha (entropy temperature) ---
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
        else:
            alpha_loss = torch.tensor(0.)

        # --- Soft update target networks ---
        if self.learn_step % self.target_update_freq == 0:
            self.soft_update(self.q1, self.q1_target)
            self.soft_update(self.q2, self.q2_target)
            logger.info("Soft-updated target networks")

        self.learn_step += 1

        logger.info(f"Update step: {self.learn_step}, Policy Loss: {policy_loss.item():.4f}, Q1 Loss: {q1_loss.item():.4f}, Q2 Loss: {q2_loss.item():.4f}, Alpha Loss: {alpha_loss.item() if isinstance(alpha_loss, torch.Tensor) else alpha_loss:.4f}, Alpha: {self.alpha:.4f}")

    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'q1_optimizer_state_dict': self.q1_optimizer.state_dict(),
            'q2_optimizer_state_dict': self.q2_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.automatic_entropy_tuning else None,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict() if self.automatic_entropy_tuning else None,
            'alpha': self.alpha,
        }, path)
        logger.info(f"SAC model saved to {path}")

    def load(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=DEVICE)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.q1.load_state_dict(checkpoint['q1_state_dict'])
            self.q2.load_state_dict(checkpoint['q2_state_dict'])
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
            self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer_state_dict'])
            self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer_state_dict'])
            if self.automatic_entropy_tuning:
                self.log_alpha = checkpoint['log_alpha']
                self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            self.alpha = checkpoint.get('alpha', self.alpha)
            logger.info(f"SAC model loaded from {path}")
        else:
            logger.warning(f"SAC model file {path} not found")