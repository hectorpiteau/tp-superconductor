
import torch.nn as nn

class MLPNet(nn.Module):
    def __init__(self, device, in_features=81) -> None:
        super().__init__()
        self.device = device
        self.fc = nn.Sequential(
                                # nn.Flatten(),
                                nn.Linear(in_features=in_features, out_features=128),
                                nn.BatchNorm1d(128),
                                nn.ReLU(),
                                nn.Linear(in_features=128, out_features=128),
                                nn.BatchNorm1d(128),
                                nn.ReLU(),
                                nn.Linear(in_features=128, out_features=64),
                                nn.BatchNorm1d(64),
                                nn.ReLU(),
                                nn.Linear(in_features=64, out_features=64),
                                nn.BatchNorm1d(64),
                                nn.Sigmoid(),
                                nn.Linear(in_features=64, out_features=32),
                                nn.BatchNorm1d(32),
                                nn.Sigmoid(),
                                nn.Linear(in_features=32, out_features=32),
                                nn.BatchNorm1d(32),
                                nn.Sigmoid(),
                                nn.Linear(in_features=32, out_features=32),
                                nn.BatchNorm1d(32),
                                nn.Sigmoid(),
                                nn.Linear(in_features=32, out_features=1),
                                nn.Sigmoid()
                                ).to(device)
    
    
    def forward(self, x):
        x = self.fc(x.to(self.device))
        return x