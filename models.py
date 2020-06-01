import torch as T
import torch.optim as optim
import torch.nn.modules as nn
import torch.nn.functional as F


class DDPGActor(nn.Module):
    """Actor network for DDPG, outputs actions given a state.

        Params
        ======
            n_states(int): number of inputs, ie. vector length of state
            n_actions(int): number of outputs, ie vector length of actions
            n_hidden(int): number of hidden neurons per layer
            lr(float): learning rate to be passed to the optimizer
            device(string): device to use for computations, 'cuda' or 'cpu'
        """
    def __init__(self, n_states, n_actions, n_hidden, lr, device):
        super(DDPGActor, self).__init__()
        self.device = device

        self.input = nn.Linear(n_states, n_hidden)
        self.l1 = nn.Linear(n_hidden, n_hidden)
        self.l2 = nn.Linear(n_hidden, n_hidden)
        self.out = nn.Linear(n_hidden, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)

    def forward(self, x):
        if not isinstance(x, T.Tensor):
            x = T.Tensor(x).unsqueeze(0).to(self.device)

        x = F.relu(self.input(x))
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.tanh(self.out(x))

        return x


class DDPGCritic(nn.Module):
    """Critic network for DDPG, outputs state-action value given a state-action input.

        Params
        ======
            n_states(int): number of inputs, ie. vector length of state
            n_actions(int): number of outputs, ie vector length of actions
            n_hidden(int): number of hidden neurons per layer
            lr(float): learning rate to be passed to the optimizer
            device(string): device to use for computations, 'cuda' or 'cpu'
        """
    def __init__(self, n_states, n_actions, n_hidden, lr, device):
        super(DDPGCritic, self).__init__()
        self.device = device

        self.input = nn.Linear(n_states+n_actions, n_hidden)
        self.l1 = nn.Linear(n_hidden, n_hidden)
        self.l2 = nn.Linear(n_hidden, n_hidden)
        self.out = nn.Linear(n_hidden, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)

    def forward(self, x, y):
        x = self.input(T.cat([x, y], dim=1))
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.out(x)

        return x
