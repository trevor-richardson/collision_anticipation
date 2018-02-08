'''
The following is a class the allows one to utilize a stateful lstm model in pytorch
This allows for the reset state to be a hyperparameter in which you can learn the most generalizable model based on how long the recurrency analysis should be
PARAMETERS
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

class StatefulConv2dLSTMCell(nn.Module):
    def __init__(self, input_shape, no_filters, kernel_shape, strides, padding='nopad', weight_init=None, reccurent_weight_init=None,  cell_weight_init=None, bias_init=None, drop=None, rec_drop=None):
        super(StatefulConv2dLSTMCell, self).__init__()

        if(weight_init==None):
            #weights need to be the shape of input or x and the output
            self.W_f = nn.Parameter(torch.zeros(kernel_shape[0], kernel_shape[1], int(input_shape[-1]), no_filters))
            self.W_f = nn.init.xavier_normal(self.W_f)
            self.W_i = nn.Parameter(torch.zeros(kernel_shape[0], kernel_shape[1], int(input_shape[-1]), no_filters))
            self.W_i = nn.init.xavier_normal(self.W_i)
            self.W_o = nn.Parameter(torch.zeros(kernel_shape[0], kernel_shape[1], int(input_shape[-1]), no_filters))
            self.W_o = nn.init.xavier_normal(self.W_o)
            self.W_c = nn.Parameter(torch.zeros(kernel_shape[0], kernel_shape[1], int(input_shape[-1]), no_filters))
            self.W_c = nn.init.xavier_normal(self.W_c)
        else:
            self.W_f = nn.Parameter(torch.zeros(kernel_shape[0], kernel_shape[1], int(input_shape[-1]), no_filters))
            self.W_f = weight_init(self.W_f)
            self.W_i = nn.Parameter(torch.zeros(kernel_shape[0], kernel_shape[1], int(input_shape[-1]), no_filters))
            self.W_i = weight_init(self.W_i)
            self.W_o = nn.Parameter(torch.zeros(kernel_shape[0], kernel_shape[1], int(input_shape[-1]), no_filters))
            self.W_o = weight_init(self.W_o)
            self.W_c = nn.Parameter(torch.zeros(kernel_shape[0], kernel_shape[1], int(input_shape[-1]), no_filters))
            self.W_c = weight_init(self.W_c)

        if(reccurent_weight_init == None):
            #Weight matrices for hidden state
            self.U_f = nn.Parameter(torch.zeros(kernel_shape[0], kernel_shape[1], no_filters, no_filters))
            self.U_f = nn.init.xavier_normal(self.U_f)
            self.U_i = nn.Parameter(torch.zeros(kernel_shape[0], kernel_shape[1], no_filters, no_filters))
            self.U_i = nn.init.xavier_normal(self.U_i)
            self.U_o = nn.Parameter(torch.zeros(kernel_shape[0], kernel_shape[1], no_filters, no_filters))
            self.U_o = nn.init.xavier_normal(self.U_o)
            self.U_c = nn.Parameter(torch.zeros(kernel_shape[0], kernel_shape[1], no_filters, no_filters))
            self.U_c = nn.init.xavier_normal(self.U_c)
        else:
            self.U_f = nn.Parameter(torch.zeros(kernel_shape[0], kernel_shape[1], no_filters, no_filters))
            self.U_f = recurrent_weight_initializer(self.U_f)
            self.U_i = nn.Parameter(torch.zeros(kernel_shape[0], kernel_shape[1], no_filters, no_filters))
            self.U_i = recurrent_weight_initializer(self.U_i)
            self.U_o = nn.Parameter(torch.zeros(kernel_shape[0], kernel_shape[1], no_filters, no_filters))
            self.U_o = recurrent_weight_initializer(self.U_o)
            self.U_c = nn.Parameter(torch.zeros(kernel_shape[0], kernel_shape[1], no_filters, no_filters))
            self.U_c = recurrent_weight_initializer(self.U_c)

        if(cell_weight_init == None):
            #Weight matrices for hidden state
            tup = (int(int(input_shape[1] - kernel_shape[0] + 1) / strides), int( int(input_shape[2] - kernel_shape[1] + 1) / strides), no_filters)
            self.V_f = nn.Parameter(torch.zeros(
                int(int(input_shape[1] - kernel_shape[0] + 1) / strides), int( int(input_shape[2] - kernel_shape[1] + 1) / strides), no_filters))
            self.V_f = nn.init.xavier_normal(self.V_f)
            self.V_i = nn.Parameter(torch.zeros(
                int( int(input_shape[1] - kernel_shape[0] + 1) / strides), int( int(input_shape[2] - kernel_shape[1] + 1) / strides), no_filters))
            self.V_i = nn.init.xavier_normal(self.V_i)
            self.V_o = nn.Parameter(torch.zeros(
                int( int(input_shape[1] - kernel_shape[0] + 1) / strides), int( int(input_shape[2] - kernel_shape[1] + 1) / strides), no_filters))
            self.V_o = nn.init.xavier_normal(self.V_o)
        else:
            self.V_f = nn.Parameter(torch.zeros(
                int( int(input_shape[1] - kernel_shape[0] + 1) / strides), int( int(input_shape[2] - kernel_shape[1] + 1) / strides), no_filters))
            self.V_f = nn.init.xavier_normal(self.V_f)
            self.V_i = nn.Parameter(torch.zeros(
                int( int(input_shape[1] - kernel_shape[0] + 1) / strides), int( int(input_shape[2] - kernel_shape[1] + 1) / strides), no_filters))
            self.V_i = nn.init.xavier_normal(self.V_i)
            self.V_o = nn.Parameter(torch.zeros(
                int( int(input_shape[1] - kernel_shape[0] + 1) / strides), int( int(input_shape[2] - kernel_shape[1] + 1) / strides), no_filters))
            self.V_o = nn.init.xavier_normal(self.V_o)

        if(bias_init==None):
            #bias initialized to zeros
            self.b_f = nn.Parameter(torch.zeros(no_filters))
            self.b_i = nn.Parameter(torch.zeros(no_filters))
            self.b_o = nn.Parameter(torch.zeros(no_filters))
            self.b_c = nn.Parameter(torch.zeros(no_filters))
        else:
            self.b_f = bias_init(torch.zeros(no_filters))
            self.b_i = bias_init(torch.zeros(no_filters))
            self.b_o = bias_init(torch.zeros(no_filters))
            self.b_c = bias_init(torch.zeros(no_filters))

        self.stride = strides
        self.kernel = kernel_shape
        self.no_filters = no_filters
        self.inp_shape = input_shape
        self.pad = padding

        #for now no padding
        self.conv2d_w = nn.Conv2d(input_shape[0], no_filters, kernel_shape, stride=strides) #Need to define the convolutional layers for pytorch
        self.conv2d_h = nn.Conv2d(no_filters, no_filters, kernel_shape, stride=strides)
        if(drop==None):
            self.dropout = nn.Dropout(0)
        else:
            self.dropout = nn.Dropout(drop)
        if(rec_drop == None):
            self.rec_dropout = nn.Dropout(0)
        else:
            self.rec_dropout = nn.Dropout(drop)

    def forward(self, X_t, previous_hidden_memory_tuple):
        h_t_previous, c_t_previous = previous_hidden_memory_tuple[0], previous_hidden_memory_tuple[1]

        X_t = self.dropout(X_t)
        h_t_previous = self.rec_dropout(h_t_previous)
        c_t_previous = self.rec_dropout(c_t_previous)

        #f(t) = sigmoid(W_f (conv) x(t) + U_f (conv) h(t-1) + V_f (*) c(t-1)  + b_f)
        f_t = F.sigmoid(
            self.conv2d_w(X_t, self.W_f, self.stride, pad=self.pad) + self.conv2d_h(h_t_previous, self.U_f, [1, 1, 1, 1]) + c_t_previous * self.V_f + self.b_f #w_f needs to be the previous input shape by the number of hidden neurons
        )
        #i(t) = sigmoid(W_i (conv) x(t) + U_i (conv) h(t-1) + V_i (*) c(t-1)  + b_i)
        i_t = F.sigmoid(
            self.conv2d_w(X_t, self.W_i, self.stride, pad=self.pad) + self.conv2d_h(h_t_previous, self.U_i, [1, 1, 1, 1]) + c_t_previous * self.V_i + self.b_i
        )
        #o(t) = sigmoid(W_o (conv) x(t) + U_o (conv) h(t-1) + V_i (*) c(t-1) + b_o)
        o_t = F.sigmoid(
            self.conv2d_w(X_t, self.W_o, self.stride, pad=self.pad) + self.conv2d_h(h_t_previous, self.U_o, [1, 1, 1, 1]) + c_t_previous * self.V_o + self.b_o
        )

        #c(t) = f(t) (*) c(t-1) + i(t) (*) hypertan(W_c (conv) x_t + U_c (conv) h(t-1) + b_c)
        c_hat_t = F.tanh(
            self.conv2d_w(X_t, self.W_c, self.stride) + self.conv2d_h(h_t_previous, self.U_c, [1, 1, 1, 1]) + self.b_c
        )
        c_t = (f_t * c_t_previous) + (i_t * c_hat_t)
        #h_t = o_t * tanh(c_t)
        h_t = o_t * F.tanh(c_t)
        #h(t) = o(t) (*) hypertan(c(t))

        return h_t, c_t

#I prefer hard sigmoid for gradient passing
# def hard_sigmoid(x):
#     """hard sigmoid for convlstm"""
#     x = (0.2 * x) + 0.5
#     zero = tf.convert_to_tensor(0., x.dtype.base_dtype)
#     one = tf.convert_to_tensor(1., x.dtype.base_dtype)
#     x = tf.clip_by_value(x, zero, one)
#     return x


'''
torch.nn.self.conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
'''
