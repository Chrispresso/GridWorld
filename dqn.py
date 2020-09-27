from typing import NamedTuple, Tuple
from collections import deque
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from config import Config


class Experience(NamedTuple):
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool

class ExperienceReplayBuffer:
    def __init__(self, buffer_size: int, batch_size: int, config: Config):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = config.DQN.device

    def add_experience(self, experience: Experience) -> None:
        self.memory.append(experience)

    def sample(self):
        samples = random.sample(self.memory, k=self.batch_size)

        # Grab S, A, R, S', Done
        # Each row is a sample
        states = np.vstack([sample.state for sample in samples])
        actions = np.vstack([sample.action for sample in samples])
        rewards = np.vstack([sample.reward for sample in samples])
        next_states = np.vstack([sample.next_state for sample in samples])
        dones = np.vstack([sample.done for sample in samples])

        # Convert the above to tensors
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).byte().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self) -> int:
        return len(self.memory)

class QNetwork(nn.Module):
    """
    https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    """
    def __init__(self, input_shape: Tuple[int, int, int], num_actions: int):
        super().__init__()
        # Get the dimensions (height, width, channels)
        h, w, c = input_shape

        # Create the network
        # 16 8x8 filters with stride 4
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=16, kernel_size=8, stride=4)
        conv1_h_out, conv1_w_out = self._get_output_shape(h, w, 8, 4)
        # 32 4x4 filters with stride 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        conv2_h_out, conv2_w_out = self._get_output_shape(conv1_h_out, conv1_w_out, 4, 2)
        # Calculate the total number of neurons in the first flattened layer
        self.num_flattened = conv2_h_out * conv2_w_out * 32
        # Create flattened layer
        self.dense1 = nn.Linear(self.num_flattened, 256)
        # Output for actions
        self.dense2 = nn.Linear(256, num_actions)

    def forward(self, state: torch.Tensor):
        # Convert from (batch size) x (height)   x (width)  x (channels) (NHWC)
        #           to (batch size) x (channels) x (height) x (width)    (NCHW)
        state = state.permute(0, 3, 1, 2).contiguous()
        x = torch.relu(self.conv1(state))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.dense1(x.reshape([-1, self.num_flattened])))
        x = self.dense2(x)
        return x


    def _get_output_shape(self, height: int, width: int, kernel_size: int, stride: int):
        """
        Gets the output height and width based on kernel size, dilation, etc.
        The only things that I consider here are height, width, kernel_size and stride.
        I keep the default values of padding and stride.

        If you need more information, look here:
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        """
        h_out = math.floor(((height - (kernel_size - 1) - 1) / stride) + 1)
        w_out = math.floor(((width - (kernel_size - 1) - 1) / stride) + 1)
        return (h_out, w_out)

class DQNAgent:
    def __init__(self, input_shape: Tuple[int, int, int], num_actions: int, config: Config):
        self._step = 0
        self.num_actions = num_actions
        self.device = config.DQN.device
        buffer_size = config.ExperienceReplay.memory_size
        batch_size = config.ExperienceReplay.batch_size
        
        self.local_net = QNetwork(input_shape, num_actions).to(self.device)
        self.target_net = QNetwork(input_shape, num_actions).to(self.device)
        self.target_net.load_state_dict(self.local_net.state_dict())

        self.tau = config.DQN.tau
        self.gamma = config.DQN.gamma
        self.soft_update_every_n = config.DQN.soft_update_every_n_episodes

        # @TODO: make the optim configurable
        self.optimizer = torch.optim.Adam(self.local_net.parameters())
        self.experience_replay = ExperienceReplayBuffer(buffer_size, batch_size, config)

    def learn(self, experiences: Tuple[Experience]) -> None:
        states, actions, rewards, next_states, dones = experiences

        state_action_vals = self.local_net(states).gather(1, actions)

        # Get state-action values for next_states assuming greedy-policy
        # unsqueeze to go from shape [batch] to [batch, 1]
        state_action_vals_next_states = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute expected
        expected_state_action_values = rewards + (self.gamma * state_action_vals_next_states * (1 - dones))

        # Clear gradient and minimize
        self.local_net.train()
        self.optimizer.zero_grad()
        #@TODO: Add loss to config
        loss = F.mse_loss(state_action_vals, expected_state_action_values)
        loss.backward()
        self.optimizer.step()

        self.soft_update()

    def soft_update(self) -> None:
        for target_param, policy_param in zip(self.target_net.parameters(), self.local_net.parameters()):
            target_param.data.copy_(self.tau*policy_param.data + (1.0-self.tau)*target_param.data)

    def step(self, experience: Experience) -> None:
        self.experience_replay.add_experience(experience)

        self._step = (self._step + 1) % self.soft_update_every_n
        if len(self.experience_replay) > 64 and self._step == 0:
            experiences = self.experience_replay.sample()
            self.learn(experiences)

        self._step += 1
    
    def act(self, state: np.ndarray, eps) -> int:
        # Convert state to [1, N] where N is the number of state dimensions
        state = torch.from_numpy(state).float().to(self.device)

        self.local_net.eval()
        with torch.no_grad():
            action_vals = self.local_net(state)
        self.local_net.train()

        if random.random() < eps:
            return random.choice(np.arange(self.num_actions))
        else:
            return np.argmax(action_vals.cpu().data.numpy())

def save_checkpoint(agent: DQNAgent, episode: int, eps: float, eps_end: float, eps_decay: float, path: str) -> None:
    torch.save({
        'episode': episode,
        'eps': eps,
        'eps_end': eps_end,
        'eps_decay': eps_decay,
        'local_model_state_dict': agent.local_net.state_dict(),
        'target_model_state_dict': agent.target_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict()
    }, path)

def load_checkpoint(agent: DQNAgent, path: str) -> Tuple[int, float, float, float]:
    checkpoint = torch.load(path)
    agent.local_net.load_state_dict(checkpoint['local_model_state_dict'])
    agent.target_net.load_state_dict(checkpoint['target_model_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return (checkpoint['episode'], checkpoint['eps'], checkpoint['eps_end'], checkpoint['eps_decay'])
