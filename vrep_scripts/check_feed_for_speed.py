#This scrip is meant to measure the speed of my feed forward pass

import numpy as np
import keras
import simplejson as json
from keras.models import model_from_json
import time
import keras.backend.tensorflow_backend as K
import tensorflow as tf

#if I use theano as backend will it be faster

def getVideo(filepath):
    vid = np.load(filepath)
    video = np.transpose(vid, (0,2,3,1))
    lst = []
    for index in range(15):
        lst.append(video[index])

    f = np.stack(lst, axis=0)
    returner = np.expand_dims(f, axis=0)
    print (returner.shape)
    return returner


def getMultipleVideos(filepath):
    vid = np.load(filepath)
    video = np.transpose(vid, (0,2,3,1))
    lst = []
    for index in range(1):
        lst.append(video[index])

    f = np.stack(lst, axis=0)
    returner = np.expand_dims(f, axis=0)
    # print (returner.shape)
    return_list = []
    for x in range(60):
        return_list.append(returner)
    return return_list

def load_deep_architecture(filepath_arch, filepath_weight):

    json_file = open(filepath_arch, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(filepath_weight)
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print("Loaded model from disk")
    return loaded_model


def getTime(model, video):

    start = time.time()
    prediction = model.predict(video, batch_size=1)
    end = time.time()
    return end - start
    #get time
    #return the difference of the two

def getDummyData(dummy_count):

    lst = []
    dummy = np.zeros((1, 1, 64, 64, 3))
    # print (dummy.shape)
    for interator in range(dummy_count - 1): #dummy count in this case represents the batch size
        lst.append(dummy)
    # print(len(lst))
    dummy_return = np.stack(lst, axis=1)
    # print ("dummy data shape   :   ", dummy_return[0].shape) # this should be the shape of the dummy data I want to concatenate to my model
    return dummy_return[0]


def getBackendMultipleVideos(model, videos):
    get_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])


    model_output = get_output([videos[0], 0])[0]

    for video in videos:
        start = time.time()
        model_output = get_output([video, 0])[0]

        end = time.time()
        print(end - start)
    print ("prediction ", model_output[0])
    return end - start


def getTimeBackend(model, video):
    print("Bypassing model.predict()")

    get_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])

    start= time.time()

    model_output = get_output([video, 0])[0]

    end = time.time()
    # print ("prediction ", model_output) # if batch size = 0
    print ("prediction ", model_output) # if batch size > 1



def main():

    '''
        Found that a batch size of 32 in stateful model is better predictor than sliding window and also has much faster prediction times. .006ms
    '''
    time = 0
    batch_size = 32 # if no batch size with the stateful model put this = 0

    model = load_deep_architecture('/home/trevor/robotics-research/deep_learning/model_size_2.json', '/home/trevor/robotics-research/deep_learning/saved_models/weights2000.h5')
    #load and preprocess the file
    # video = getVideo('/home/trevor/robotics-research/data_generated/ran_test/hit/3138collision_video.npy') #single video prediction
    video = getMultipleVideos('/home/trevor/robotics-research/data_generated/ran_test/hit/3138collision_video.npy') #multiple video prediction

    if batch_size > 0: # if I need the inputs to be batched
        dummy_data = getDummyData(batch_size)
        for index in range(len(video)):
            video[index] = np.concatenate((video[index], dummy_data), axis =0)

    getBackendMultipleVideos(model, video)

    print ("The amount of time it took to do my forward pass : ", time)


if __name__ == '__main__':
    main()
