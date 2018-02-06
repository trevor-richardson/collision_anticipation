import vrep
import sys
import time
import numpy as np
from scipy.misc import imsave
import random
import scipy.io as sio
import scipy
import keras
import simplejson as json
from keras.models import model_from_json
from multiprocessing.pool import ThreadPool
import tensorflow as tf
from os.path import isfile, join
from os import listdir
import os
import configparser

config = configparser.ConfigParser()
config.read('../config.ini')

dir_path = config['DEFAULT']['BASE_DIR']

'''
-Make pyQt graphs live with this demo

To run this code make sure you use the following in order to load hit and miss or replay the last video
python3 stateful_demo.py hit
python3 stateful_demo.py miss
python3 stateful_demo.py replay

This demo is meant for a stateful model. In order to appease a stateful model fake data must be generated because the input size the stateful model desires is (32, 1, 64, 64, 3)
.013s
77 HZ  -- about the image collection speed
'''

dummy_count = 32 #just the batch size
pool = ThreadPool(processes=1)
'''
The following loads my model for predictions
'''
json_file = open(dir_path + '/machine_learning/saved_models/stateful_model/current_version/current_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
# with tf.device('/gpu:0'):
loaded_model.load_weights(dir_path + '/machine_learning/saved_models/stateful_model/current_version/weights9067.h5')
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
graph = tf.get_default_graph()

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

'''
This is a global list that the asynchronous thread uses to make predictions and store for the main process to analyze after each timestep
dummy_data is the np.zeros which is required for my models prediction -- tensor of shape (31, 1, 64, 64, 3)
'''

lst = []
dummy = np.zeros((1, 1, 64, 64, 3))
# print (dummy.shape)
for interator in range(dummy_count - 1): #dummy count in this case represents the batch size
    lst.append(dummy)
dummy_return = np.stack(lst, axis=1)
# print ("dummy data shape   :   ", dummy_return[0].shape) # this should be the shape of the dummy data I want to concatenate to my model
dummy_data = dummy_return[0]
pain_values_list = []

def start():
    makeFirstPrediction(np.zeros((32, 1, 64, 64, 3)), -10) #make a first prediction which takes  the most time in prediction
    vrep.simxFinish(-1) # just in case, close all opened connections
    clientID=vrep.simxStart('127.0.0.1',19997,True,True,5000,5) #start my Connection
    #x =vrep.simxStopSimulation(clientID,vrep.simx_opmode_oneshot)
    error_code =vrep.simxStartSimulation(clientID,vrep.simx_opmode_oneshot_wait)
    return clientID, error_code

'''
This function makes a prediction
'''
def makePrediction(video, timestep):
    global graph

    with graph.as_default():
        prediction = loaded_model.predict_on_batch(video)
        # line.set_data(timestep/10 - 13, prediction[0][0])
        # plt.draw()
        print (int(timestep/10)," " , prediction[0])
        pain_values_list.append(prediction[0])

def makeFirstPrediction(video, timestep):
    global graph

    with graph.as_default():
        start = time.time()
        prediction = loaded_model.predict_on_batch(video)
        loaded_model.reset_states()
        end = time.time()
        # line.set_data(timestep/10 - 13, prediction[0][0])
        # plt.draw()
        # print("first prediction")
        # print (int(timestep/10)," " , prediction[0], end - start)
        print("##################################")

def runModel(clientID, current_iteration):
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
                # scipy.misc.imsave(str(count) + 'outfile.png', rotate_img)
                if image_count % rate_image_collection == 0: #changing this may help
                    # str_name =  str(current_iteration) + 'collision_video' + str(count)
                    # scipy.misc.imsave(str(image_count) + '.png', rotate_img) #save the video or not
                    # print("\t\t\t collecting an image")
                    input_to_model = np.expand_dims(rotate_img, axis=0)
                    input_to_model = np.expand_dims(input_to_model, axis=0)
                    # print("input to model   ", input_to_model.shape, "     dummy_data    ", dummy_data.shape)
                    video_to_model = np.concatenate((input_to_model, dummy_data), axis=0)
                    ret = pool.apply_async(makePrediction, (video_to_model, image_count))
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
        return list_of_images
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
    # pool.close()

    with open('coll_classification.txt', 'a') as f:
        f.write('%d\n' % detector[1])
        f.close()
    if detector[1] == 1:
        print ("\nHit")
        return 1
    else:
        print ("\nMiss")
        return 0


def single_simulation(n_iter):
    if sys.argv[1] == 'hit':
        load_hit_or_miss_npy(True) #change this boolean value to pull from hits or misses
    elif sys.argv[1] == 'miss':
        load_hit_or_miss_npy(False)
    clientID, start_error = start()
    runModel(clientID, n_iter)
    collision_signal = detectCollisionSignal(clientID) #This records whether hit or miss
    end_error = end(clientID)

def main(iter_start, iter_end):
    #load keras model for prediction once
    # loaded_model = load_deep_architecture('/deep_learning/saved_models/small_model_1/model_ran.json', '/deep_learning/saved_models/small_model_1/weights006.h5')
    #run n number of trials for demo purposes
    for current_iteration in range(iter_start, iter_end):
        single_simulation(current_iteration)
    pool.close()

if __name__ == '__main__':
    main(0,1)
