"""
Model for Question 2 of hwk3.
@author: Samuel Lavoie
"""
from torch import nn


class Critic(nn.Module):
    # DONT MODIFY. Use this module for every questions of Q1. Your tests might fail if you modify this.
    def __init__(self, in_size=2):
        super(Critic, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_size, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, input):
        output = self.main(input)
        return output.squeeze()
