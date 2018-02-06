
import sys
import time
import numpy as np
import random
import simplejson as json
from os.path import isfile, join
from os import listdir
import matplotlib.pyplot as plt
from matplotlib import cm, colors

class VisualizeActivations(object):
    def __init__(self, act_list, movie):

        self.layer = 0
        self.stride = 5
        self.activation_list = act_list #shape should be list of [next0, next1, next2] -- next0 should be shape 64, 2, 32, 32, 20
        self.h_or_c = 1 #1 if I want c

        print("initializing visualizer")
        self.video = movie
        if self.layer == 0:
            self.num_row = 6
            self.num_column = 6
        elif self.layer == 1:
            self.num_row = 7
            self.num_column = 5
        elif self.layer == 2:
            self.num_row = 6
            self.num_column = 7
        else:
            print("\n\ndesired layer is incorrect\n\n")
            sys.exit()

    def visualize_activation(self):
        act_list = []
        count = 0
        prev_var =0
        for element in self.activation_list:
            # print (element[0].shape , "should be batch size 1, 2, 32, 32, 20")
            var = element[self.layer][0][self.h_or_c]
            # print("length", len(var[0][0]))
            for filter_no in range(len(var[0][0])):
                var2 = (var[:, :, filter_no])
                var2 = 255*var2
                act_list.append(var2)
            self.plot_activations(act_list, count, self.video[count])
            prev_var = var
            act_list = []
            count +=1

    def plot_activations(self, activations_list, timestep, original_image):
        fig = plt.figure(1)
        if self.h_or_c == 0:
            fig.suptitle('Activations for Layer ' +  str(self.layer) + ' at Time ' + str(timestep) , fontsize=14, fontweight='bold')
        else:
            fig.suptitle('Cell State ' +  str(self.layer) + ' at Time ' + str(timestep) , fontsize=14, fontweight='bold')
        fig.set_size_inches(100,100)
        count = 0
        stride_count = 1
        plt.subplot(self.num_row, self.num_column, 1)
        plt.imshow(original_image)
        plt.title('Original Image')
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_ticks([])
        frame1.axes.get_yaxis().set_ticks([])
        if timestep % self.stride == 0:
            for activation in activations_list:
                plt.subplot(self.num_row, self.num_column, count+2)
                plt.imshow(activation, cmap=cm.gray)
                plt.title('Convolution '+ str(count))
                frame1 = plt.gca()
                frame1.axes.get_xaxis().set_ticks([])
                frame1.axes.get_yaxis().set_ticks([])
                count+=1
            fig.subplots_adjust(hspace=.5)
            frame1 = plt.gca()
            for xlabel_i in frame1.axes.get_xticklabels():
                xlabel_i.set_fontsize(0.0)
            for xlabel_i in frame1.axes.get_yticklabels():
                xlabel_i.set_fontsize(0.0)
            plt.show()
