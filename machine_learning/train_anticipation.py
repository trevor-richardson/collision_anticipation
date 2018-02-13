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

import argparse
import configparser

config = configparser.ConfigParser()
config.read('../config.ini')

base_dir = config['DEFAULT']['BASE_DIR']
from data_generator import VideoDataGenerator
from anticipation_model import Custom_Spatial_Temporal_Anticipation_NN

'''Argparse Variables for Dynamic Experimentation

Set up all of my arg parser arguments --
Need to finish training and test/val
Need to visualize outputs
Need to allow for padding
Need to implement demo with current system
Need to update readme and take a video
Need to post open to github

'''
#for argparser to handle bool
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Collision AnticipationModel in Pytorch')

#training and testing args
parser.add_argument('--exp_iteration', type=int, default=0, metavar='N',
                    help='For saving the best model based on validation of one experiment')

parser.add_argument('--lr', type=float, default=.0001,
                    help='The learning rate used for my Adam optimizer (default: .0001)')

args = parser.parse_args()
'''Define my model'''

model = Custom_Spatial_Temporal_Anticipation_NN((3, 64, 64), (32, 24, 16), (5, 5), 2, .3, 1)
if args.cuda:
    model.cuda()
    print("Using GPU acceleration")
optimizer = optim.Adam(model.parameters(), lr=args.lr)

'''Training'''


'''Testing/Validation'''


'''Helper Function'''

def print_parameters(params_list):
    total = 0
    for element in params_list:
        adder = 1
        for inner_element in element.size():
            adder*=inner_element
        print(element.size())
        total+=adder
    print("Total number of paramters in model:", total)


def view_image(image, name):
    plt.imshow(image.numpy(), cmap='gray')
    plt.title(name)
    plt.show()

'''Initialize experiment'''
def main():
    print("Starting program")
    training_directory = base_dir + '/data_generated/image_only/train/'
    validation_directory = base_dir + '/data_generated/image_only/val/'
    testing_directory = base_dir + '/data_generated/image_only/test/'
    train_number_of_miss = 1137
    train_number_of_hit = 1137
    val_number_of_miss = 131
    val_number_of_hit = 131
    test_number_of_miss = 132
    test_number_of_hit = 132
    print_parameters(model.parameters())

    generator = VideoDataGenerator(training_directory, validation_directory,
    testing_directory, train_number_of_hit, train_number_of_miss, val_number_of_hit,
        val_number_of_miss, test_number_of_hit, test_number_of_miss)
    train, train_class, val, val_class, test, test_class = generator.prepare_data()

if __name__ == '__main__':
    main()
