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
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = dir_path.split('robotics-research')[0]

pool = ThreadPool(processes=1)
'''
The following loads my model for predictions
'''
json_file = open('/home/twrichar/robotics_research/deep_learning/saved_models/small_model_1/model_ran.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights('/home/twrichar/robotics_research/deep_learning/saved_models/small_model_1/weights006.h5')
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
graph = tf.get_default_graph()
'''
This is a global list that the asynchronous thread uses to make predictions and store for the main process to analyze after each timestep
'''
pain_values_list = []

def start():
    makeFirstPrediction(np.zeros((1, 15, 64, 64, 3)), 23) #make a first prediction which takes  the most time in prediction
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
        prediction = loaded_model.predict(video, batch_size=1)
        # line.set_data(timestep/10 - 13, prediction[0][0])
        # plt.draw()
        print (int(timestep/10 - 14)," " , prediction[0][0])
        pain_values_list.append(prediction[0][0])

def makeFirstPrediction(video, timestep):
    global graph
    with graph.as_default():
        start = time.time()
        prediction = loaded_model.predict(video, batch_size=1)
        end = time.time()
        # line.set_data(timestep/10 - 13, prediction[0][0])
        # plt.draw()
        print("first prediction")
        print (int(timestep/10 - 14)," " , prediction[0][0], end - start)

def runModel(clientID, current_iteration):
    list_of_images = []
    check_my_pain = 0
    ret_code, left_handle = vrep.simxGetObjectHandle (clientID,'Pioneer_p3dx_leftMotor', vrep.simx_opmode_oneshot_wait)
    ret_code, right_handle = vrep.simxGetObjectHandle (clientID,'Pioneer_p3dx_rightMotor', vrep.simx_opmode_oneshot_wait)
    move_bool=False

    if clientID!=-1:
        # ret, sphere_handle = vrep.simxGetObjectHandle(clientID,'Sphere',vrep.simx_opmode_oneshot_wait)
        res,v0=vrep.simxGetObjectHandle(clientID,'Vision_sensor',vrep.simx_opmode_oneshot_wait)
        res,resolution,image=vrep.simxGetVisionSensorImage(clientID,v0,0,vrep.simx_opmode_streaming)
        collision_bool = False
        count = 0

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
                if count % 10 == 0: #changing this may help
                    # str_name =  str(current_iteration) + 'collision_video' + str(count)
                    scipy.misc.imsave(str(count) + '.png', rotate_img)
                    list_of_images.append(rotate_img) #append a movie to the current sliding window

                if len(list_of_images) == 15:
                    # scipy.misc.imsave(str(count) + 'outfile.png', rotate_img)
                    input_to_model = np.stack(list_of_images, axis=0)
                    input_to_model = np.expand_dims(input_to_model, axis=0)
                    ret = pool.apply_async(makePrediction, (input_to_model, count))
                    check_my_pain +=1
                    list_of_images.pop(0) #delete last movie in sliding window
                    # print ("length biiiiizzz ", len(list_of_images), count)
                    if (check_my_pain > 10 and move_bool==False):
                        recent_pain_values = pain_values_list[-8:]
                        if(min(recent_pain_values) > .5 ):
                            print("predicting hit " , check_my_pain, min(recent_pain_values))
                            move_bool = True
                            moveRobot(clientID, "left", left_handle, right_handle)


                count+=1
        return list_of_images
    else:
        sys.exit()

#close VREP
def end(clientID):
    #end and cleanup
    error_code =vrep.simxStopSimulation(clientID,vrep.simx_opmode_oneshot_wait)
    vrep.simxFinish(clientID)
    return error_code

#check if there was a collision by calling globale variable set inside VREP
def detectCollisionSignal(clientID):
    detector = 0
    collision_str = "collision_signal"
    detector = vrep.simxGetIntegerSignal(clientID, collision_str, vrep.simx_opmode_oneshot_wait)
    pool.close()
    pool.terminate()
    with open('coll_classification.txt', 'a') as f:
        f.write('%d\n' % detector[1])
        f.close()
    if detector[1] == 1:
        print ("\nHit")
        return 1
    else:
        print ("\nMiss")
        return 0

def moveRobot(clientID, direction, left_handle, right_handle):
    #get robot handler
    #set robot motion to hard right or hard left depending on the
    if(direction==None):
        print("staying put")
        vrep.simxSetJointTargetVelocity(clientID, left_handle, 0, vrep.simx_opmode_oneshot)
        vrep.simxSetJointTargetVelocity(clientID, right_handle, 0, vrep.simx_opmode_oneshot)
    else:
        if(direction=='right'):
            print("going right")
            vrep.simxSetJointTargetVelocity(clientID, right_handle, 30, vrep.simx_opmode_oneshot)
            vrep.simxSetJointTargetVelocity(clientID, left_handle, 30, vrep.simx_opmode_oneshot)
            vrep.simxSetJointTargetVelocity(clientID, right_handle, 30, vrep.simx_opmode_oneshot)
            vrep.simxSetJointTargetVelocity(clientID, left_handle, 30, vrep.simx_opmode_oneshot)
        else:
            print ("going left")
            vrep.simxSetJointTargetVelocity(clientID, left_handle, 100, vrep.simx_opmode_oneshot_wait)
            vrep.simxSetJointTargetVelocity(clientID, right_handle, 100, vrep.simx_opmode_oneshot_wait)

def single_simulation(n_iter):
    clientID, start_error = start()
    runModel(clientID, n_iter)
    collision_signal = detectCollisionSignal(clientID) #This records whether hit or miss
    end_error = end(clientID)

def main(iter_start, iter_end):
    #load keras model for prediction once
    # loaded_model = load_deep_architecture('/home/twrichar/robotics_research/deep_learning/saved_models/small_model_1/model_ran.json', '/home/twrichar/robotics_research/deep_learning/saved_models/small_model_1/weights006.h5')
    #run n number of trials for demo purposes
    for current_iteration in range(iter_start, iter_end):
        single_simulation(current_iteration)

if __name__ == '__main__':
    main(0,1)
