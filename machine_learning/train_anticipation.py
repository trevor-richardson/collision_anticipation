import time
import numpy as np
import random
import math
import sys
import scipy.io as sio
import os
from os.path import isfile, join
from os import listdir
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F

from anticipation_model import Custom_Spatial_Temporal_Anticipation_NN


def main():
    print("Starting program")
    model = Custom_Spatial_Temporal_Anticipation_NN((3, 64, 64), 32, (5, 5), 2, .3, 1)


if __name__ == '__main__':
    main()
