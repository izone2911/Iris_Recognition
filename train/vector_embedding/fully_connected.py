from torch import nn

class FullyConnected(nn.Module):
    def __init__(self, input_dim, output_dim=128, margin=1.0, *args, **kwargs):
        # dense layer
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.fc3 = nn.Linear(output_dim, output_dim)
        self.rl = nn.ReLU()
        self.margin = margin

    def forward(self, x):
        x = self.fc1(x)
        x = self.rl(x)
        x = self.fc2(x)
        x = self.rl(x)
        x = self.fc3(x)
        return x