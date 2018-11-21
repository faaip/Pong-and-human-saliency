# Visualizing and Understanding Atari Agents | Sam Greydanus | 2017 | MIT License

from __future__ import print_function
#import warnings ; warnings.filterwarnings('ignore') # mute warnings, live dangerously ;)

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

import glob
import numpy as np
from scipy.misc import imresize # preserves single-pixel info _unlike_ img = img[::2,::2]

class NNPolicy(torch.nn.Module): # an actor-critic neural network
    def __init__(self, channels, num_actions, memsize):
        super(NNPolicy, self).__init__()
        self.memsize = memsize
        self.conv1 = nn.Conv2d(channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.lstm = nn.LSTMCell(32 * 5 * 5, self.memsize)
        self.critic_linear, self.actor_linear = nn.Linear(self.memsize, 1), nn.Linear(self.memsize, num_actions)

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = x.view(-1, 32 * 5 * 5)
        hx, cx = self.lstm(x, (hx, cx))
        return self.critic_linear(hx), self.actor_linear(hx), (hx, cx)

    def try_load(self, save_dir, checkpoint='*.tar'):
        paths = glob.glob(save_dir + checkpoint) ; step = 0
        if len(paths) > 0:
            ckpts = [int(s.split('.')[-2]) for s in paths]
            ix = np.argmax(ckpts) ; step = ckpts[ix]
            self.load_state_dict(torch.load(paths[ix]))
        print("\tno saved models") if step is 0 else print("\tloaded model: {}".format(paths[ix]))
        return step

'''
class NNPolicy(torch.nn.Module): # an actor-critic neural network
    def __init__(self, num_actions, channels, memsize):
        super(NNPolicy, self).__init__()
        self.conv1 = nn.Conv2d(channels, 4, 3, stride=2, padding=1, bias = None)
        self.conv2 = nn.Conv2d(4, 8, 3, stride=2, padding=1, bias = None)
        self.conv3 = nn.Conv2d(8, 16, 3, stride=2, padding=1, bias = None)
        self.conv4 = nn.Conv2d(16, 32, 3, stride=2, padding=1, bias = None)
        self.gru = nn.GRUCell(32 * 5 * 5, memsize, bias = None)
        self.critic_linear, self.actor_linear = nn.Linear(memsize, 1, bias = None), nn.Linear(memsize, num_actions, bias = None)

    def forward(self, inputs, train=True, hard=False):
        inp, hx = inputs
        x = F.elu(self.conv1(inp))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        hx = self.gru(x.view(-1, 32 * 5 * 5), hx)
        return self.critic_linear(hx), self.actor_linear(hx), hx

    def try_load(self, save_dir):
        paths = glob.glob('/Users/maximoldenbeek/Documents/ITandCognition/2.1-CS/Maxim/Model Exploration/pong-v4/*.tar') ; step = 0
        if len(paths) > 0:
            ckpts = [int(s.split('.')[-2]) for s in paths]
            ix = np.argmax(ckpts) ; step = ckpts[ix]
            self.load_state_dict(torch.load(paths[ix]))
        print("\tno saved models") if step is 0 else print("\tloaded model: {}".format(paths[ix]))
        return step
'''