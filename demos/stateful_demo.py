import vrep
import sys
import time
import numpy as np
from scipy.misc import imsave
import random
import scipy.io as sio
import scipy
import simplejson as json
from multiprocessing.pool import ThreadPool
from os.path import isfile, join
from os import listdir
import os
import configparser
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

config = configparser.ConfigParser()
config.read('../config.ini')

dir_path = config['DEFAULT']['BASE_DIR'] #path given in the config file
sys.path.append(dir_path + '/machine_learning/')

from anticipation_model import Custom_Spatial_Temporal_Anticipation_NN
parser = argparse.ArgumentParser(description='Collision AnticipationModel in Pytorch')

parser.add_argument('--kernel_0', type=int, default=5, metavar='N',
                    help='Kernel for layer 0 (default 5)')

parser.add_argument('--strides', type=int, default=2, metavar='N',
                    help='Strides for the convolutions in the convlstm layers (default 2)')

parser.add_argument('--model_path', default='89.44754464285714.pth', type=str,
                    help='This is type of expirement I want to run (train, test, visualize)')

parser.add_argument('--no_filters_0', type=int, default=40, metavar='N',
                    help='Number of activation maps to in layer 0 (default 40)')

parser.add_argument('--no_filters_1', type=int, default=30, metavar='N',
                    help='Number of activation maps to in layer 1 (default 30)')

parser.add_argument('--no_filters_2', type=int, default=20, metavar='N',
                    help='Number of activation maps to in layer 2 (default 20)')

parser.add_argument('--drop_rte', type=float, default=.3, metavar='N',
                    help='dropout rate (default .3)')
#Demo specific global variables
parser.add_argument('--num_runs', type=int, default=15, metavar='N',
                    help='This represents the number of times to run the simulation in the background (default 15)')

args = parser.parse_args()

def create_lstm_states(shape, batch_size):
    c = Variable(torch.zeros(batch_size, shape[0], shape[1], shape[2])).float().cuda()
    h = Variable(torch.zeros(batch_size, shape[0], shape[1], shape[2])).float().cuda()
    return (h, c)

'''
This demo is meant for a stateful model.
Speed of inference is critical
'''
pool = ThreadPool(processes=1)
'''
The following loads my model for predictions
'''
rgb_shape = (3, 64, 64)

model = Custom_Spatial_Temporal_Anticipation_NN(rgb_shape, (args.no_filters_0,
    args.no_filters_1, args.no_filters_2), (args.kernel_0, args.kernel_0), args.strides, 1,
    padding=0, dropout_rte=args.drop_rte)

if torch.cuda.is_available():
    model.cuda()
    print("Using GPU acceleration")
try:
    model.load_state_dict(torch.load(dir_path + "/machine_learning/saved_models/" + args.model_path))
except ValueError:
    print("Not a valid model to load")
    sys.exit()

'''
This is a global list that the asynchronous thread uses to make predictions and store for the main process to analyze after each timestep
dummy_data is the np.zeros which is required for my models prediction -- tensor of shape (31, 1, 64, 64, 3)
'''
pain_values_list = []

def start():
    vrep.simxFinish(-1) # just in case, close all opened connections
    clientID=vrep.simxStart('127.0.0.1',19997,True,True,5000,5) #start my Connection
    #x =vrep.simxStopSimulation(clientID,vrep.simx_opmode_oneshot)
    error_code =vrep.simxStartSimulation(clientID,vrep.simx_opmode_oneshot_wait)
    return clientID, error_code

def makePrediction(video, timestep, states):
    model.eval()
    output, states = model(video, states)
    pain_values_list.append(output[0].data[0])
    return states

def makeFirstPrediction(video, timestep, states):
    model.eval()
    output, states = model(video, states)
    return states

def runModel(clientID, current_iteration, states):
    list_of_images = []
    check_my_pain = 0
    rate_image_collection = 10
    ret_code, left_handle = vrep.simxGetObjectHandle (clientID,'DynamicLeftJoint', vrep.simx_opmode_oneshot_wait)
    ret_code, right_handle = vrep.simxGetObjectHandle (clientID,'DynamicRightJoint', vrep.simx_opmode_oneshot_wait)

    if clientID!=-1:
        # ret, sphere_handle = vrep.simxGetObjectHandle(clientID,'Sphere',vrep.simx_opmode_oneshot_wait)
        res,v0=vrep.simxGetObjectHandle(clientID,'Vision_sensor',vrep.simx_opmode_oneshot_wait)
        res,resolution,image=vrep.simxGetVisionSensorImage(clientID,v0,0,vrep.simx_opmode_streaming)
        collision_bool = False
        pain_bool = True
        image_count = 0

        t_end = time.time() + 2
        t_start = time.time()
        while (vrep.simxGetConnectionId(clientID)!=-1 and time.time() < t_end):
            res,resolution,image=vrep.simxGetVisionSensorImage(clientID,v0,0,vrep.simx_opmode_buffer)
            if res==vrep.simx_return_ok:
                img = np.array(image,dtype=np.uint8)
                img.resize([resolution[1],resolution[0],3])
                rotate_img = img.copy()
                rotate_img = np.flipud(img)
                if image_count % rate_image_collection == 0: #changing this may help
                    video_to_model = np.transpose(np.expand_dims(rotate_img, axis=0), (0, 3, 1, 2))
                    data = torch.from_numpy(np.flip(video_to_model,axis=0).copy()).float()
                    if torch.cuda.is_available():
                        data = data.cuda()
                    data = Variable(data, volatile=True)
                    if image_count !=0:
                        states = ret.get() #get the async return from last iteration
                    ret = pool.apply_async(makePrediction, (data, image_count, states))
                    #The following just checks the recent 8 pain values and reasons based on the minimum value
                    if (check_my_pain > 14 and pain_bool):
                        recent_pain_values = pain_values_list[-8:]
                        if(min(recent_pain_values) > .5 ):
                            pain_bool = False
                            print("predicting hit " , check_my_pain, min(recent_pain_values))
                            if np.random.uniform() < .5:
                                return_val = vrep.simxSetJointTargetVelocity(clientID, left_handle, 50, vrep.simx_opmode_oneshot)
                                return_val2 = vrep.simxSetJointTargetVelocity(clientID, right_handle, 50, vrep.simx_opmode_oneshot_wait)
                            else:
                                return_val = vrep.simxSetJointTargetVelocity(clientID, left_handle, -50, vrep.simx_opmode_oneshot)
                                return_val2 = vrep.simxSetJointTargetVelocity(clientID, right_handle, -50, vrep.simx_opmode_oneshot_wait)

                    if(not pain_bool):
                        recent_pain_values = pain_values_list[-8:]
                        if (min(recent_pain_values) < .5):
                            pain_bool = True
                            print("IM SAFE NOW")
                            return_val = vrep.simxSetJointTargetVelocity(clientID, left_handle, 0, vrep.simx_opmode_oneshot)
                            return_val2 = vrep.simxSetJointTargetVelocity(clientID, right_handle, 0, vrep.simx_opmode_oneshot_wait)
                    check_my_pain+=1
                image_count+=1
    else:
        sys.exit()

#close VREP
def end(clientID):
    #end and cleanup
    error_code =vrep.simxStopSimulation(clientID,vrep.simx_opmode_oneshot_wait)
    vrep.simxFinish(clientID)
    return error_code

'''
Hits
Load pos and velo for a hit video
Right now I'm running on training data for the hits
'''
def load_hit_or_miss_npy(hit_bool):

    txt_files = []
    read_hit = dir_path + '/vrep_scripts/saved_vel_pos_data/test/hit/'
    read_miss = dir_path + '/vrep_scripts/saved_vel_pos_data/test/miss/'
    write_pos = dir_path + '/demos/vrep_pos_velo/pos.txt'
    write_velo = dir_path + '/demos/vrep_pos_velo/velo.txt'

    if hit_bool:
        print("loading hit")
        txt_files = [f for f in listdir(read_hit) if isfile(join(read_hit, f))]
        for indx, element in enumerate(txt_files):
            txt_files[indx] = read_hit + element
    else:
        print("loading miss")
        txt_files = [f for f in listdir(read_miss) if isfile(join(read_miss, f))]
        for indx, element in enumerate(txt_files):
            txt_files[indx] = read_miss + element

    #load a random file
    element = random.choice(txt_files) # get the random file to load
    print(element)
    info = np.load(element)

    position = (info[0], info[1], info[2])
    velocity = (info[3], info[4], info[5])

    #write the info too write_pos and write_velo
    with open(write_pos, "w") as wr_pos:
        print(position[0], file=wr_pos)
        print(position[1], file=wr_pos)
        print(position[2], file=wr_pos)

    with open(write_velo, "w") as wr_velo:
        print(velocity[0], file=wr_velo)
        print(velocity[1], file=wr_velo)
        print(velocity[2], file=wr_velo)

#check if there was a collision by calling globale variable set inside VREP
def detectCollisionSignal(clientID):
    detector = 0
    collision_str = "collision_signal"
    detector = vrep.simxGetIntegerSignal(clientID, collision_str, vrep.simx_opmode_oneshot_wait)
    with open('coll_classification.txt', 'a') as f:
        f.write('%d\n' % detector[1])
        f.close()
    if detector[1] == 1:
        print ("\nHit")
        return 1
    else:
        print ("\nMiss")
        return 0

def single_simulation(n_iter, states, hit_or_miss):
    if hit_or_miss:
        load_hit_or_miss_npy(True) #change this boolean value to pull from hits or misses
    else:
        load_hit_or_miss_npy(False)
    clientID, start_error = start()
    runModel(clientID, n_iter, states)
    collision_signal = detectCollisionSignal(clientID) #This records whether hit or miss
    end_error = end(clientID)

def main(iter_start, iter_end):
    prev0 = create_lstm_states(model.convlstm_0.output_shape, 1)
    prev1 = create_lstm_states(model.convlstm_1.output_shape, 1)
    prev2 = create_lstm_states(model.convlstm_2.output_shape, 1)
    states = [prev0, prev1, prev2]
    data = torch.from_numpy(np.zeros((1, 3, 64, 64))).float()
    if torch.cuda.is_available():
        data = data.cuda()
    data = Variable(data, volatile=True)
    tim = time.time()
    states = makeFirstPrediction(data, -10, states)
    for current_iteration in range(0, args.num_runs):
        prev0 = create_lstm_states(model.convlstm_0.output_shape, 1)
        prev1 = create_lstm_states(model.convlstm_1.output_shape, 1)
        prev2 = create_lstm_states(model.convlstm_2.output_shape, 1)
        states = [prev0, prev1, prev2]
        hit_or_miss = random.randint(0, 1)
        single_simulation(current_iteration, states, hit_or_miss)
    pool.close()

if __name__ == '__main__':
    main(0,1)
