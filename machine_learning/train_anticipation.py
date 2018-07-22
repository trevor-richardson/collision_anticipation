import time
import numpy as np
import random
import math
import sys
import scipy.io as sio
import os
from os.path import isfile, join
from os import listdir

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import argparse
import configparser

config = configparser.ConfigParser()
config.read('../config.ini')

base_dir = config['DEFAULT']['BASE_DIR']
from data_generator import VideoDataGenerator
from anticipation_model import Custom_Spatial_Temporal_Anticipation_NN
from visualizer import VisualizeActivations

'''Argparse Variables for Dynamic Experimentation

Need to allow for padding !!!!!!!!!!!!
Need to update readme and take a video
Need to post open to github
'''

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Collision AnticipationModel in Pytorch')

#training and testing args
parser.add_argument('--exp_iteration', type=int, default=64, metavar='N',
                    help='Batch size default size is 64')

parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='Number of hit and number of miss videos (default 64)')

parser.add_argument('--h_or_c', type=int, default=0, metavar='N',
                    help='Zero if I want to visualize h_t 1 if I want to visualize c_t (default 0)')

parser.add_argument('--view_model_params', type=str2bool, nargs='?', default=False,
                    help='This variable stores outputs and targets in a list to bp all at once (default: True)')

parser.add_argument('--view_hit', type=str2bool, nargs='?', default=True,
                    help='Default is True which means a hit video')

parser.add_argument('--exp_type', default='train', type=str,
                    help='This is type of expirement I want to run (train, test, visualize)')

parser.add_argument('--visualize_str', default='h_t', type=str,
                    help='If we are visualizing the hidden activations or cell state use this')

parser.add_argument('--model_to_load', default='89.44754464285714', type=str,
                    help='This is the model of interest to load')

parser.add_argument('--num_epochs', type=int, default=50, metavar='N',
                    help='Number of hit and number of miss videos (default 50)')

parser.add_argument('--no_filters_0', type=int, default=40, metavar='N',
                    help='Number of activation maps to in layer 0 (default 40)')

parser.add_argument('--no_filters_1', type=int, default=30, metavar='N',
                    help='Number of activation maps to in layer 1 (default 30)')

parser.add_argument('--no_filters_2', type=int, default=20, metavar='N',
                    help='Number of activation maps to in layer 2 (default 20)')

parser.add_argument('--kernel_0', type=int, default=5, metavar='N',
                    help='Kernel for layer 0 (default 5)')

parser.add_argument('--kernel_1', type=int, default=5, metavar='N',
                    help='Kernel for layer 1 (default 5)')

parser.add_argument('--kernel_2', type=int, default=5, metavar='N',
                    help='Kernel for layer 2 (default 5)')

parser.add_argument('--padding_0', type=int, default=2, metavar='N',
                    help='Padding for layer 0 (default 2)')

parser.add_argument('--padding_1', type=int, default=2, metavar='N',
                    help='Padding for layer 1 (default 2)')

parser.add_argument('--padding_2', type=int, default=2, metavar='N',
                    help='Padding for layer 2 (default 2)')

parser.add_argument('--lr', type=float, default=.0001,
                    help='The learning rate used for my Adam optimizer (default: .0001)')

parser.add_argument('--strides', type=int, default=2, metavar='N',
                    help='Strides for the convolutions in the convlstm layers (default 2)')

parser.add_argument('--drop_rte', type=float, default=.3, metavar='N',
                    help='dropout rate (default .3)')

parser.add_argument('--output_shape', type=int, default=1, metavar='N',
                    help='output_shape (default 1)')

parser.add_argument('--no_train_vid', type=int, default=1137, metavar='N',
                    help='Number of hit and number of miss videos (default 1137)')

parser.add_argument('--no_val_vid', type=int, default=131, metavar='N',
                    help='Number of hit and number of miss videos (default 131)')

parser.add_argument('--no_test_vid', type=int, default=132, metavar='N',
                    help='Number of hit and number of miss videos (default 132)')

args = parser.parse_args()


'''Define my model'''
rgb_shape = (3, 64, 64)
model = Custom_Spatial_Temporal_Anticipation_NN(rgb_shape, (args.no_filters_0,
    args.no_filters_1, args.no_filters_2), (args.kernel_0, args.kernel_0), args.strides, 1,
    padding=0, dropout_rte=args.drop_rte)
if torch.cuda.is_available():
    model.cuda()
    print("Using GPU acceleration")
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

'''Training'''
def train_model(epoch, data_files, label):
    model.train()

    tim = time.time()
    predicted_list = []
    y_list = []
    train_loss = 0
    train_accuracy = 0
    train_step_counter = 0

    for index in range(int(len(data_files)/args.batch_size)):
        current_video = load_next_batch(data_files[index*args.batch_size:(index+1)*args.batch_size])
        current_label = np.asarray(label[index*args.batch_size:(index+1)*args.batch_size])
        target = torch.from_numpy((current_label))
        if torch.cuda.is_available():
            target = target.cuda()
        target = Variable(target)

        prev0 = create_lstm_states(model.convlstm_0.output_shape, args.batch_size)
        prev1 = create_lstm_states(model.convlstm_1.output_shape, args.batch_size)
        prev2 = create_lstm_states(model.convlstm_2.output_shape, args.batch_size)
        states = [prev0, prev1, prev2]
        optimizer.zero_grad()

        for inner_index in range(int(current_video.shape[0])):
            data = torch.from_numpy(current_video[inner_index]).float()
            if torch.cuda.is_available():
                data = data.cuda()
            data = Variable(data)

            output, states = model(data, states)
            predicted_list.append(output)
            y_list.append(target)

        pred = torch.cat(predicted_list)
        y_ = torch.cat(y_list).float()
        loss = F.binary_cross_entropy(pred, y_)

        loss.backward()
        optimizer.step()
        train_loss+=loss.data
        train_step_counter +=1

        del(predicted_list[:])
        del(y_list[:])

    print("Training time for one epoch", time.time() - tim)
    # print(loss.data[0], train_loss.cpu().numpy()[0], train_loss.cpu().numpy()[0]/train_step_counter)
    print('Train Epoch: {}\tLoss: {:.6f}'.format(
        epoch, train_loss.cpu().numpy()[0]/train_step_counter))
    return train_loss.cpu().numpy()[0]/train_step_counter


'''Testing/Validation'''
def test_model(data_files, label):
    model.eval()

    tim = time.time()
    test_loss = 0
    correct = 0
    instance_counter = 0
    test_step_counter = 0

    for index in range(int(len(data_files)/args.batch_size)):
        current_video = load_next_batch(data_files[index*args.batch_size:(index+1)*args.batch_size])
        current_label = np.asarray(label[index*args.batch_size:(index+1)*args.batch_size])
        target = torch.from_numpy((current_label))
        if torch.cuda.is_available():
            target = target.cuda()
        target = Variable(target.float())

        prev0 = create_lstm_states(model.convlstm_0.output_shape, args.batch_size)
        prev1 = create_lstm_states(model.convlstm_1.output_shape, args.batch_size)
        prev2 = create_lstm_states(model.convlstm_2.output_shape, args.batch_size)
        states = [prev0, prev1, prev2]

        for inner_index in range(int(current_video.shape[0])):
            data = torch.from_numpy(current_video[inner_index]).float()
            if torch.cuda.is_available():
                data = data.cuda()
            data = Variable(data, volatile=True)

            output, states = model(data, states)
            test_loss += F.binary_cross_entropy(output, target, size_average=False).data[0] # sum up batch loss
            pred = torch.round(output.data) # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum()
            instance_counter+=1


    test_loss /= (instance_counter * args.batch_size)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
        test_loss, correct, instance_counter * args.batch_size,
        100. * correct / (instance_counter * args.batch_size)))
    return 100. * correct / (instance_counter * args.batch_size)


'''Visualize Activations'''

def visualize_learning(data_files, label, view_hit):
    model.eval()
    index = 0
    if view_hit:
        for element in label:
            if element == 1:
                break
            index+=1
    else:
        for element in label:
            if element == 0:
                break
            index+=1
    tim = time.time()
    test_loss = 0
    correct = 0
    instance_counter = 0
    test_step_counter = 0
    activation_list = []
    current_video = load_next_batch(data_files[index:(index+1)])
    current_label = np.asarray(label[index:(index+1)])

    target = torch.from_numpy((current_label))

    if torch.cuda.is_available():
        target = target.cuda()
    target = Variable(target.float())

    prev0 = create_lstm_states(model.convlstm_0.output_shape, 1)
    prev1 = create_lstm_states(model.convlstm_1.output_shape, 1)
    prev2 = create_lstm_states(model.convlstm_2.output_shape, 1)
    states = [prev0, prev1, prev2]

    for inner_index in range(int(current_video.shape[0])):
        data = torch.from_numpy(current_video[inner_index]).float()
        if torch.cuda.is_available():
            data = data.cuda()
        data = Variable(data, volatile=True)

        output, states = model(data, states)

        visualize_0 = np.transpose(np.asarray([states[0][0].data.cpu().numpy(), states[0][1].data.cpu().numpy()]),(1, 0, 3, 4, 2))
        visualize_1 = np.transpose(np.asarray([states[1][0].data.cpu().numpy(), states[1][1].data.cpu().numpy()]),(1, 0, 3, 4, 2))
        visualize_2 = np.transpose(np.asarray([states[2][0].data.cpu().numpy(), states[2][1].data.cpu().numpy()]),(1, 0, 3, 4, 2))

        activation_list.append([visualize_0, visualize_1, visualize_2])

        test_loss += F.binary_cross_entropy(output, target, size_average=False).data[0] # sum up batch loss
        pred = torch.round(output.data) # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).sum()
        instance_counter+=1



    visualizer = VisualizeActivations(activation_list, np.squeeze(current_video), args.h_or_c)
    visualizer.visualize_activation()

    test_loss /= (instance_counter * args.batch_size)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
        test_loss, correct, instance_counter * args.batch_size,
        100. * correct / (instance_counter * args.batch_size)))
    return 100. * correct / (instance_counter * args.batch_size)


'''Helper Functions'''
def create_lstm_states(shape, batch_size):
    c = Variable(torch.zeros(batch_size, shape[0], shape[1], shape[2])).float().cuda()
    h = Variable(torch.zeros(batch_size, shape[0], shape[1], shape[2])).float().cuda()
    return (h, c)


def load_next_batch(string_names):
    lst = []
    for element in string_names:
    #list of string names
        data = np.load(element)
        lst.append(data)
    movies = np.stack(lst, axis =0)
    movies = np.transpose(movies, (1, 0, 2, 3, 4))
    return movies


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


def save_model(model, acc):
    torch.save(model.state_dict(), base_dir + '/machine_learning/saved_models/' + str(acc) + '.pth')


def load_model(path):
    global model
    try:
        model.load_state_dict(torch.load(base_dir + "/machine_learning/saved_models/" + path + ".pth"))
    except ValueError:
        print("Not a valid model to load")
        sys.exit()


'''Initialize experiment'''
def main():
    global model
    if args.view_model_params:
        print_parameters(model.parameters())

    training_directory = base_dir + '/data_generated/image_only/train/'
    validation_directory = base_dir + '/data_generated/image_only/val/'
    testing_directory = base_dir + '/data_generated/image_only/test/'

    generator = VideoDataGenerator(training_directory, validation_directory,
    testing_directory, args.no_train_vid, args.no_val_vid, args.no_test_vid)

    train, train_class, val, val_class, test, test_class = generator.prepare_data()
    accuracies = []
    if args.exp_type == 'train':
        print("Training")
        best_acc = 0.0
        for index in range(args.num_epochs):
            tr_acc = train_model(index, train, train_class)
            acc = test_model(val, val_class)
            accuracies.append([tr_acc, acc])
            print("\n\n**************************************************\n")
            if acc > best_acc:
                best_acc = acc
                save_model(model, acc)
    elif args.exp_type == 'test':
        print("Loading and Testing")
        load_model(args.model_to_load)
        acc = test_model(test, test_class)
    else:
        print("Visualizing")
        load_model(args.model_to_load)
        visualize_learning(test, test_class, args.view_hit)
    np.save("graph_acc", np.asarray(accuracies))

if __name__ == '__main__':
    main()
