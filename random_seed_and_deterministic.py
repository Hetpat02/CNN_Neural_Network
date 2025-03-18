# -------- create folder/directory in your code --------

import os

if not os.path.exists("./model/"):
    os.makedirs("./model/")


# Set a random seed for everything and make your training process deterministic and reproducible

# Without setting the rand seed, you will get different random number everytime you run your code.
import torch

print(torch.rand(1,3))

#------------

import random, os
import numpy as np
import torch

seed = 1000

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.
# benchmark mode is good whenever your input sizes for your network do not vary. This way, cudnn will look for the optimal set of algorithms for that particular configuration (which takes some time). This usually leads to faster runtime.
# But if your input sizes changes at each iteration, then cudnn will benchmark every time a new size appears, possibly leading to worse runtime performances.

print(torch.rand(1,3))
print(torch.rand(1,3))

# ------- an example in your training code -----
if args.rand_seed:
    seed = random.randint(1, 999)
    print("Using random seed:", seed)
else:
    seed = args.seed
    print("Using manual seed:", seed)
    
np.random.seed(seed)
torch.manual_seed(seed)
if args.cuda:
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==================================================
# ------ example of using argparser package ---------

import argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR training')

# Neural Network settings
parser.add_argument('--arch', type=str, default=None,
                    help='[vgg, resnet, convnet, alexnet]')
parser.add_argument('--depth', default=None, type=int,
                    help='depth of the neural network, 16,19 for vgg; 18, 50 for resnet')

# data settings
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--dataset', type=str, default="cifar10",
                    help='[cifar10， cifar100]')

# optimizer setting
parser.add_argument('--optmzr', type=str, default='SGD', metavar='OPTMZR',
                    help='optimizer used [SGD, adam] (default: SGD)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

# lr schedule
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--lr-decay', type=int, default=60, metavar='LR_decay',
                    help='how many every epoch before lr drop (default: 60)')
parser.add_argument('--lr-scheduler', type=str, default='default',
                    help='define lr scheduler [cosine, step]')

# training setting
parser.add_argument('--rand-seed', action='store_true', default=False,
                    help='use random seed')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()

print(args.arch)
