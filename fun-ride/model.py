import numpy as np
class NN_Model():
    def __init__(self,n_bots,track_lenght,tracksegment_lenght,n_inner_neurons,intial_velocity,max_height ,intial_height,intial_track_angle):
        self.track_lenght = track_lenght
        self.tracksegment_lenght = tracksegment_lenght
        self.n_inner_neurons = n_inner_neurons
        self.intial_velocity = intial_velocity
        self.max_height = max_height
        self.intial_height = intial_height
        self.intial_track_angle = intial_track_angle
        self.n_bots = n_bots
        
        
        
        
        
        self.g = 9.81
        self.n_tracksegments = track_lenght//tracksegment_lenght

        self.norm_velocity = (intial_velocity^2)/(2*g*max_height)
        self.norm_height = intial_height/max_height
        self.norm_energy = 1
        self.norm_track_angle = intial_track_angle/180

        self.input_vector = [norm_velocity,norm_height,norm_energy,norm_track_angle]

        self.generation_output = np.zeros((len(input_vector),n_bots,n_tracksegments))
        self.next_input = np.zeros((len(input_vector),n_bots,n_tracksegments))
        self.x_nodes = np.zeros((n_tracksegments+1,n_bots))
        self.y_nodes = np.zeros((n_tracksegments+1,n_bots))
        self.y_nodes[0,:]= intial_height

        self.y_splines = np.zeros((n_tracksegments+1,n_bots))
        self.y_splines[0,:] = intial_height

        self.track_angles = np.zeros((n_tracksegments+1,n_bots))
        self.track_angles[0,:] = norm_track_angle*np.pi

        self.velocities = np.zeros((n_tracksegments+1,n_bots))
        self.velocities[0,:] = intial_velocity

        self.y_splines_deri_1 = np.zeros((n_tracksegments+1,n_bots))
        self.y_splines_deri_2 = np.zeros((n_tracksegments+1,n_bots))
        self.roc = np.zeros((n_tracksegments+1,n_bots))
        self.centripetal_acc = np.zeros((n_tracksegments+1,n_bots))
        self.g_force = np.zeros((n_tracksegments+1,n_bots))

        self.weights_1 = np.zeros(len(input_vector),n_inner_neurons,n_bots)
        self.weights_2 = np.zeros((n_inner_neurons,n_inner_neurons,n_bots))
        self.weights_3 = np.zeros((n_inner_neurons,1,n_bots))

        self.bias_1 = np.zeros((1,n_inner_neurons,n_bots))
        self.bias_2 = np.zeros((1,n_inner_neurons,n_bots))
        for i in range(n_bots):
            for j in range(n_tracksegments):
                self.next_input[:,i,j]=self.input_vector
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

    def compute_track_angles(self,i,j,k):
        delta_track_angle = np.tanh(self.next_input[:,j,k].T*weights_1[:,:,j]+bias_1[:,:,j])
        delta_track_angle = np.tanh(delta_track_angle*weights_2[:,:,j]+bias_2[:,:,j])
        delta_track_angle = np.tanh(delta_track_angle*weights_3[:,:,j])
        track_angle = self.track_angles[k,j]/np.pi + delta_track_angle/(10/self.tracksegment_lenght)
        self.track_angles[k+1,j] = track_angle *np.pi
        return track_angle
    def compute_x_y_nodes(self,i,j,k,track_angle):
        delta_x = self.tracksegment_lenght*np.cos(track_angle*np.pi)
        delta_y = self.tracksegment_lenght*np.sin(track_angle*np.pi)
        self.x_nodes[k+1,j] = self.x_nodes[k,j]+delta_x
        self.y_nodes[k+1,j] = self.y_nodes[k,j]+delta_y
        return delta_x,delta_y
    def compute_generation_outputs(self,i,j,k,track_angle,delta_y,delta_energy):
        self.generation_output[1,j,k] = self.next_input[1,j,k]*(2*self.g*self.max_height)-2*(self.g*delta_y+delta_energy)
        self.generation_output[2,j,k] = self.next_input[2,j,k]*self.max_height+delta_y
        self.generation_output[3,j,k] = self.next_input[3,j,k]*self.g*self.max_height-delta_energy
        self.generation_output[4,j,k] = track_angle*np.pi
    def update_inputs(self,i,j,k):
        self.next_input[1,j,k+1] = self.generation_output[1,j,k]/(2*self.g*self.max_height)
        self.next_input[2,j,k+1] = self.generation_output[2,j,k]/self.max_height
        self.next_input[3,j,k+1] = self.generation_output[3,j,k]/(self.g*self.max_height)
        self.next_input[4,j,k+1] = self.generation_output[4,j,k]/np.pi
    def run(self,n_generations):
        for i in range(n_generations):
            for j in range(self.n_bots):
                delta_energy = 0
                for k in range(self.n_tracksegments):
                    track_angle = self.compute_track_angles(i,j,k)
                    delta_x,delta_y = self.compute_x_y_nodes(i,j,k,track_angle)
                    self.compute_generation_outputs(i,j,k,track_angle,delta_y,delta_energy)
                    self.update_inputs(i,j,k)
                    


