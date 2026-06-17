# class to make a siamese network
import torch
import torch.nn as nn

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # CNN layer
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3,64,(10,10)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2)),

            nn.Conv2d(64,128,(7,7)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2)),

            nn.Conv2d(128,128,(4,4)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4,4)),
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(9216,4096)

    def forward_once(self, x):
        x = self.cnn1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x

    def forward(self, anchor, positive, negative):
        return self.forward_once(anchor), self.forward_once(positive), self.forward_once(negative)