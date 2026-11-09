"""
This file structures the siamese network

There will be three classes, one is an embedding network that vectorizes images,
the other class is a triplet net wrapper. 
The wrapper takes three images and then passes it through the embedding layer and then returning it.
The tripletloss class returns 
"""

import torch
import torch.nn as nn

class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)
    
class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    # takes the euclidian distance from the positive and anchor as well as the euclidian distance from the negative and anchor. 
    # the loss is just 0 if the distance from the positive to the anchor is smaller than the distance from the negative to the anchor with a margin variable
    # this makes it so that the anchor is how much you want the positive vectors to be closer to the anchor than the negative
    def forward(self, anchor, positive, negative):
        pos_distance = ((anchor - positive) ** 2).sum(dim=1)
        neg_distance = ((anchor - negative) ** 2).sum(dim=1)
        loss = torch.relu(pos_distance - neg_distance + self.margin)
        return loss.mean()