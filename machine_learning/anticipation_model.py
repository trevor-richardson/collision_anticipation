from conv_lstm_cell import StatefulConv2dLSTMCell
import torch
import torch.nn as nn
import torch.nn.functional as F

'''Define the neural network'''
class Custom_Spatial_Temporal_Anticipation_NN(nn.Module):
    def __init__(self, input_shp, no_filters, kernel_size, strides, dropout_rte, output_shp):
        super(Custom_Spatial_Temporal_Anticipation_NN, self).__init__()
        print("Running Custom Built Stateful Conv2dLSTM")

        self.convlstm_0 = StatefulConv2dLSTMCell(input_shp, no_filters, kernel_size, strides)
        need_to_calc_shp0 = (32, 30, 30) #this should be a global variable in the output shape

        self.convlstm_1 = StatefulConv2dLSTMCell(need_to_calc_shp0, no_filters, kernel_size, strides)
        need_to_calc_shp1 = (32, 10, 10)

        self.convlstm_2 = StatefulConv2dLSTMCell(need_to_calc_shp1, no_filters, kernel_size, strides)
        need_to_calc = 5

        self.dropout = nn.Dropout(dropout_rte)
        self.fcn1 = nn.Linear(need_to_calc, output_shp)

    def forward(self, x, states):

        hx_0, cx_0 = self.convlstm_0(x, (states[0][0] ,states[0][1]))
        hx_1, cx_1 = self.convlstm_1(hx_0, (states[1][0] ,states[1][1]))
        hx_2, cx_2 = self.convlstm_2(hx_1, (states[2][0] ,states[2][1]))

        dropped = self.dropout(hx_2)
        x = self.fcn1(dropped)
        return x, [[hx_0, cx_0], [hx_1, cx_1], [hx_2, cx_2]]
