
import torch.nn as nn

class MLPNet(nn.Module):
    def __init__(self, device, in_features=81) -> None:
        super().__init__()
        self.device = device
        self.fc = nn.Sequential(nn.Linear(in_features=in_features, out_features=in_features),
                                nn.ReLU(),
                                nn.Linear(in_features=in_features, out_features=in_features),
                                nn.ReLU(),
                                nn.Linear(in_features=in_features, out_features=1)).to(device)
    
    
    def forward(self, x):
        x = self.fc(x.to(self.device))
        return x