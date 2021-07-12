from .nn_parameters import track_lenght,tracksegment_lenght,n_inner_neurons
from .coaster_parameters import intial_velocity,max_height ,intial_height,intial_track_angle
import numpy as np
g = 9.81
n_tracksegments = track_lenght//tracksegment_lenght

norm_velocity = (intial_velocity^2)/(2*g*max_height)
norm_height = intial_height/max_height
norm_energy = 1
norm_track_angle = intial_track_angle/180

input_vector = [norm_velocity,norm_height,norm_energy,norm_track_angle]

generation_output = np.zeros((len(input_vector),n_bots,n_tracksegments))
next_input = np.zeros((len(input_vector),n_bots,n_tracksegments))
x_nodes = np.zeros((n_tracksegments+1,n_bots))
y_nodes = np.zeros((n_tracksegments+1,n_bots))
y_nodes[0,:]= intial_height

y_splines = np.zeros((n_tracksegments+1,n_bots))
y_splines[0,:] = intial_height

track_angles = np.zeros((n_tracksegments+1,n_bots))
track_angles[0,:] = norm_track_angle*np.pi

velocities = np.zeros((n_tracksegments+1,n_bots))
velocities[0,:] = intial_velocity

y_splines_deri_1 = np.zeros((n_tracksegments+1,n_bots))
y_splines_deri_2 = np.zeros((n_tracksegments+1,n_bots))
roc = np.zeros((n_tracksegments+1,n_bots))
centripetal_acc = np.zeros((n_tracksegments+1,n_bots))
g_force = np.zeros((n_tracksegments+1,n_bots))

weights_1 = np.zeros(len(input_vector),n_inner_neurons,n_bots)
weights_2 = np.zeros((n_inner_neurons,n_inner_neurons,n_bots))
weights_3 = np.zeros((n_inner_neurons,1,n_bots))

bias_1 = np.zeros((1,n_inner_neurons,n_bots))
bias_2 = np.zeros((1,n_inner_neurons,n_bots))

def intialize_next_input(next_input,input_vector):
    for i in range(n_bots):
        for j in range(n_tracksegments):
            next_input[:,i,j]=input_vector
    return next_input

def intialize_weights_biases(weights_1,weights_2,weights_3,bias_1,bias_2):
    for i in range(n_bots):
        for j in range(len(input_vector)):
            for k in range(n_inner_neurons):
                weights_1[j,k,i] = 2*np.random.random()-1

        for j in range(n_inner_neurons):
            for k in range(n_inner_neurons):
                weights_2[j,k,i] = 2*np.random.random()-1
        for j in range(n_inner_neurons):
            weights_3[j,1,i] = 2*np.random.random()-1
            bias_1[1,j,i] = 2*np.random.random()-1
            bias_2[1,j,i] = 2*np.random.random()-1
    return weights_1,weights_2,weights_3,bias_1,bias_2