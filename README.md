# Robot Pain Anticipation

The following is a proof of concept and simulation implementation of a spatio-temporal
machine learning deep neural network that predicts future collisions before they occur.

A custom built deep [Convolutional LSTM](https://arxiv.org/pdf/1506.04214.pdf), implemented in
[PyTorch](http://pytorch.org/) is used to predict future collisions with an object placed in
projectile motion. The projectile is set in projectile motion with an intial random velocity in
the x, y and z direction.

Trained on 3000 hit and miss simulations. -- Input to the neural network is a (64, 64, 3) image
Validated and tested on over 600 other randomly generated simulations.

Specific contributions
* Custom Built ConvLSTM Cell Class
'''
check out conv_lstm_cell.py, anticipation_model.py
'''
* "Dodgeball" robotic simulation in [V-REP](http://www.coppeliarobotics.com/)
'''
check out demo.ttt
'''
* Visualization Class that can view the activations or cell state (what's been learned) of ConvLSTM
'''
check out visualizer.py
'''
* Data generator that doesn't crash your RAM by loading filepaths and just in time producing video tensors (70, 64, 64, 3)
### Installing

Packages needed to run the code
numpy
scipy
python3
pytorch
matplotlib
vrep

I used ubuntu 16.04

In order to make the code work change base_dir in config.ini to absolute path where /collision_anticipation exists

In the vrep_scenes directory both the demo.ttt and current_scene.ttt have lua code written for the sphere object that
is custom and there are filepaths in both that need to be changed in order to run -- both need to point at the vrep_scripts folder

### Scripts to run

If properly installed, and demo.ttt is loaded in V-REP one can dodge balls with the script:

  python3 stateful_demo.py

One can visualize activations by running:

  python3 train_anticipation.py --exp_type=activations

One can train new models by:

  python3 train_anticipation.py


One collect data by running the following script with current_scene.ttt loaded in V-REP:

  python3 run_vrep_simulation.py
