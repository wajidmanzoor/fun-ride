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

        self.input_vector = [self.norm_velocity,self.norm_height,self.norm_energy,self.norm_track_angle]

        self.generation_output = np.zeros((len(self.input_vector),n_bots,self.n_tracksegments))
        self.next_input = np.zeros((len(self.input_vector),n_bots,self.n_tracksegments))
        self.x_nodes = np.zeros((self.n_tracksegments+1,n_bots))
        self.y_nodes = np.zeros((self.n_tracksegments+1,n_bots))
        self.y_nodes[0,:]= intial_height

        self.y_splines = np.zeros((self.n_tracksegments+1,n_bots))
        self.y_splines[0,:] = intial_height

        self.track_angles = np.zeros((self.n_tracksegments+1,n_bots))
        self.track_angles[0,:] = self.norm_track_angle*np.pi

        self.velocities = np.zeros((self.n_tracksegments+1,n_bots))
        self.velocities[0,:] = intial_velocity

        self.y_splines_deri_1 = np.zeros((self.n_tracksegments+1,n_bots))
        self.y_splines_deri_2 = np.zeros((self.n_tracksegments+1,n_bots))
        self.roc = np.zeros((self.n_tracksegments+1,n_bots))
        self.centripetal_acc = np.zeros((self.n_tracksegments+1,n_bots))
        self.g_force = np.zeros((self.n_tracksegments+1,n_bots))

        self.weights_1 = np.zeros(len(self.input_vector),n_inner_neurons,n_bots)
        self.weights_2 = np.zeros((n_inner_neurons,n_inner_neurons,n_bots))
        self.weights_3 = np.zeros((n_inner_neurons,1,n_bots))

        self.bias_1 = np.zeros((1,n_inner_neurons,n_bots))
        self.bias_2 = np.zeros((1,n_inner_neurons,n_bots))
        self.score = np.zeros((self.n_bots,1))
        self.score_breakdown = np.zeros((11,self.n_bots))
        self.loop_up = np.zeros((self.n_bots,1))
        self.loop_down = np.zeros((self.n_bots,1))

        f"""or i in range(n_bots):
            for j in range(self.n_tracksegments):
                self.next_input[:,i,j]=self.input_vector
        for i in range(n_bots):
            for j in range(len(self.input_vector)):
                for k in range(n_inner_neurons):
                    self.weights_1[j,k,i] = 2*np.random.random()-1

            for j in range(n_inner_neurons):
                for k in range(n_inner_neurons):
                    self.weights_2[j,k,i] = 2*np.random.random()-1
            for j in range(n_inner_neurons):
                self.weights_3[j,0,i] = 2*np.random.random()-1
                self.bias_1[0,j,i] = 2*np.random.random()-1
                self.bias_2[0,j,i] = 2*np.random.random()-1 """ 
        self.weights_1 = 2*np.random.uniform(size=self.weights_1.shape)-1
        self.weights_2 =  2*np.random.uniform(size=self.weights_2.shape)-1
        self.weights_3 =  2*np.random.uniform(size=self.weights_3.shape)-1
        
        self.bias_1 =  2*np.random.uniform(size=self.bias_1.shape)-1
        self.bias_2 =  2*np.random.uniform(size=self.bias_2.shape)-1



    def compute_track_angles(self,i,j,k):
        delta_track_angle = np.tanh(self.next_input[:,j,k].T*self.weights_1[:,:,j]+self.bias_1[:,:,j])
        delta_track_angle = np.tanh(delta_track_angle*self.weights_2[:,:,j]+self.bias_2[:,:,j])
        delta_track_angle = np.tanh(delta_track_angle*self.weights_3[:,:,j])
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
        self.generation_output[0,j,k] = self.next_input[0,j,k]*(2*self.g*self.max_height)-2*(self.g*delta_y+delta_energy)
        self.generation_output[1,j,k] = self.next_input[1,j,k]*self.max_height+delta_y
        self.generation_output[2,j,k] = self.next_input[2,j,k]*self.g*self.max_height-delta_energy
        self.generation_output[3,j,k] = track_angle*np.pi
    def update_inputs(self,i,j,k):
        self.next_input[0,j,k+1] = self.generation_output[0,j,k]/(2*self.g*self.max_height)
        self.next_input[1,j,k+1] = self.generation_output[1,j,k]/self.max_height
        self.next_input[2,j,k+1] = self.generation_output[2,j,k]/(self.g*self.max_height)
        self.next_input[3,j,k+1] = self.generation_output[3,j,k]/np.pi
    def compute_segment_radius(self,i,j,k):
        delta_1_i = np.tan(self.track_angles[k,j])
        delta_1_f = np.tan(self.track_angles[k+1,j])
        delta_2_i = (delta_1_f-delta_1_i)/(self.x_nodes[k+1,j]-self.x_nodes[k,j])
        if self.x_nodes[k+1,j] > self.x_nodes[k,j]:
            if abs(delta_1_i) < 20:
                radius = ((1+delta_1_i^2)^(3/2))/delta_2_i
            else:
                radius = np.inf
        else:
            if abs(delta_1_i) < 20:
                radius = -((1+delta_1_i^2)^(3/2))/delta_2_i
            else:
                radius = np.inf
        return radius

    def compute_energy_loss(self,i,j,k,radius,drag_coeff,friction_coeff):
        centri_acc = self.generation_output[0,j,k]/(self.g*radius)
        normal_force = abs(radius+np.cos(self.track_angles[k,j]))
        delta_energy = (friction_coeff*normal_force+drag_coeff*0.6125*self.generation_output[0,j,k]/10000)*self.tracksegment_lenght
        return delta_energy

    def compute_roc(self,i,j):
        self.roc[0,j] = np.inf
        self.roc[self.n_tracksegments+1,j] = np.inf
        for k in range(1,self.n_tracksegments):
            self.y_splines_deri_2[k,j]= (np.tan(self.track_angles[k+1,j])-np.tan(self.track_angles[k,j]))/(0.5(self.x_nodes[k+1,j]-self.x_nodes[k-1,j]))
            a = np.sqrt((self.x_nodes[k,j]-self.x_nodes[k-1,j])^2 + (self.y_nodes[k,j]-self.y_nodes[k-1,j])^2)
            b = np.sqrt((self.x_nodes[k+1,j]-self.x_nodes[k,j])^2 + (self.y_nodes[k+1,j]-self.y_nodes[k,j])^2)
            c = np.sqrt((self.x_nodes[k+1,j]-self.x_nodes[k-1,j])^2 + (self.y_nodes[k+1,j]-self.y_nodes[k-1,j])^2)
            s = 0.5*(a+b+c)
            tar = np.sqrt(s*(s-a)*(s-b)*(s-c))
            if self.x_nodes[k,j]>self.x_nodes[k-1,j]:
                if self.y_splines_deri_2[k,j]>0:
                    self.roc[k,j] = (a*b*c)/(4*tar)
                else:
                    self.roc[k,j] = -(a*b*c)/(4*tar)
            else:
                if self.y_splines_deri_2[k,j]>0:
                    self.roc[k,j] = -(a*b*c)/(4*tar)
                else:
                    self.roc[k,j] = (a*b*c)/(4*tar)
        for k in range(1,self.n_tracksegments):
            if self.roc[k-1,j] >0 and self.roc[k+1,j]>0 and self.roc[k,j]<0:
                self.roc[k,j]=-1*self.roc[k,j]
            elif self.roc[k-1,j] <0 and self.roc[k+1,j]<0 and self.roc[k,j]>0:
                self.roc[k,j]=-1*self.roc[k,j]
    def compute_velocities(self,i,j):
        for k in range(self.n_tracksegments):
            self.velocities[k+1,j]=self.generation_output[0,j,k]^(1/2)
    def compute_gforce(self,i,j):
        for k in range(self.n_tracksegments+1):
            self.g_force = self.centripetal_acc[k,j]+np.cos(self.track_angles[k,j])
    def penalty_for_moving_sideways(self,i,j,k,inversion,intial):
        if self.x_nodes[k+1,j] < self.x_nodes[k,j] and inversion is False:
            self.score[j,0] -= 1
            self.score_breakdown[0,j] -= 1
        if self.x_nodes[k,j] < -self.intial_height/2  and inversion is True:
            self.score[j,1] -= 100
            self.score_breakdown -= 100
    
    def update_inversion_score(self,i,j,k,inversion,g_force):
        '''g_force tuple specifying range (min,max)'''
        if self.x_nodes[k+1,j] < self.x_nodes[k,j] and self.x_nodes[k+1,j] > 0 and self.y_nodes[k+1,j] < self.max_height and self.y_splines_deri_1[k,j] < 0 and self.roc[k,j] > 0 and self.g_force[k,j] <=  g_force[1] and self.g_force[k,j] >= g_force[0] and self.loop_up[j,0] == 0 and inversion is True:
            self.score[j,0] +=9000
            self.score_breakdown[1,j] +=9000
            self.loop_up[j,0] = 1 
        if self.x_nodes[k+1,j] < self.x_nodes[k,j] and self.x_nodes[k+1,j]  > 0 and self.y_nodes[k+1,j] < self.max_height and self.y_splines_deri_1[k,j] > 0 and self.roc[k,j] > 0 and self.g_force[k,j] <= g_force[1] and self.g_force[k,j] >= g_force[0] and self.loop_up[j,0] > 0 and self.loop_down[j,0] ==0 and inversion is True:
            self.score[j,0] +=1000
            self.score_breakdown[1,j] += 1000
            self.loop_down[j,0] = 1 

    def penalty_for_height_voilation(self,i,j,k,min_height,inversion):
        if self.y_splines_deri_1[k,j] < min_height:
            if inversion:
                self.score[j,0] -=100
                self.score_breakdown[2,j] -= 100
            else:
                self.score[j,0] -= 1
                self.score_breakdown -= 1
    def update_gforce_score(self,i,j,k,g_force,gpositive,gnegative,inversion):
        if self.g_force[k,j] <= g_force[1] and self.g_force[k,j] >= g_force[0]:
            if self.g_force[k,j] > 0:
                self.score[j,0] += gpositive*abs(self.g_force[k,j]/(2*g_force[1]))
                self.score_breakdown[3,j] +=  gpositive*abs(self.g_force[k,j]/(2*g_force[1]))
            else:
                self.score[j,0] += gnegative*abs(self.g_force[k,j]/g_force[0])
                self.score_breakdown[4,j] += gnegative*abs(self.g_force[k,j]/g_force[0])
        elif self.g_force[k,j] > g_force[1]:
            if inversion:
                self.score[j,0]-=100
                self.score_breakdown[5,j] -= 100
            else:
                self.score[j,0] -= 1
                self.score_breakdown[5,j] -= 1
        else:
            if inversion:
                self.score[j,0] -= 100
                self.score_breakdown[6,j] -= 100
            else:
                self.score[j,0] -= 1
                self.score_breakdown[6,j] -= 1
    
    def update_velocity_score(self,i,j,k,inversion,alpha):
        if np.isreal(self.velocities[k,j]):
            self.score[j,0] += alpha*(self.velocities[k,j]/(2*self.g*self.max_height))
            self.score_breakdown[8,j] += alpha*(self.velocities[k,j]/(2*self.g*self.max_height))
        else:
            if inversion:
                self.score[j,0] -= 100
                self.score_breakdown[7,j] -= 100
            else:
                self.score[j,0] -= 1
                self.score_breakdown[7,j] -= 1

    def penalty_for_velocity_voilation(self,i,j,k,min_drop_velocity,inversion):
        if self.velocities[k,j] > min_drop_velocity and self.intial_velocity == 0:
            self.intial_velocity =1
        if self.velocities[k,j] < min_drop_velocity and self.intial_velocity == 1:
            if inversion:
                self.score[j,0]-=100
                self.score_breakdown[9,j] -= 100
            else:
                self.score[j,0] -= 1
                self.score_breakdown[9,j] -= 1
    def sort_normalize_score(self,n=5,verbose=True):
        sorted_score , sorted_ind = np.sort(self.score,axis=None)[::-1],np.argsort(self.score,axis=None)[::-1]
        if np.sum(self.loop_up[sorted_ind[0:min(self.n_bots,n)],0]) >0 or np.sum(self.loop_down[sorted_ind[0:min(self.n_bots,n)],0]) >0:
            sorted_score /=10100
        else:
            sorted_score /=10
        if verbose:
            for i in range(min(self.n_bots,n)):
                print(f'{i} Top Score: {sorted_score[i]}')
        return sorted_score,sorted_ind

    def update_steepness_score(self,i,j,k,beta):
        self.score[j,0] += beta*abs((self.y_splines[k+1,j]-self.y_splines[k,j])/(5*self.tracksegment_lenght))
        self.score_breakdown[10,j] += beta*abs((self.y_splines[k+1,j]-self.y_splines[k,j])/(5*self.tracksegment_lenght))

    def store_nn_for_top_bots(self,sorted_ind,prob):
        top_taken = max(1,int(prob*self.n_bots))
        for j in range(top_taken):
            self.weights_1[:,:,j] = self.weights_1[:,:,sorted_ind[j,0]]
            self.weights_2[:,:,j] = self.weights_2[:,:,sorted_ind[j,0]]
            self.weights_3[:,:,j] = self.weights_3[:,:,sorted_ind[j,0]]
 
            self.bias_1[:,:,j] = self.bias_1[:,:,sorted_ind[j,0]]
            self.bias_2[:,:,j] = self.bias_2[:,:,sorted_ind[j,0]]
        for j in range(1,5):
            self.weights_1[:,:,j*top_taken:(j+1)*top_taken] = self.weights_1[:,:,0:top_taken]
            self.weights_2[:,:,j*top_taken:(j+1)*top_taken] = self.weights_2[:,:,0:top_taken]
            self.weights_3[:,:,j*top_taken:(j+1)*top_taken] = self.weights_2[:,:,0:top_taken]

            self.bias_1[:,:,j*top_taken:(j+1)*top_taken] = self.bias_1[:,:,0:top_taken]
            self.bias_2[:,:,j*top_taken:(j+1)*top_taken] = self.bias_2[:,:,0:top_taken]
        

    def mutate_nn(self,percent_1,percent_2,mutate_prob,max_mutate_percent):
        taken = (max(1,int(percent_1*self.n_bots)),int(percent_2*self.n_bots))
        '''for j in range(taken[0],taken[1]):
            for k in range(self.weights_1.shape[0]):
                for z in range(self.weights_1.shape[1]):
                    if np.random.random() < mutate_prob:
                        self.weights_1[k,z,j] += (2*max_mutate_percent*np.random.random()-max_mutate_percent)
            for k in range(self.weights_2.shape[0]):
                for z in range(self.weights_2.shape[1]):
                    if np.random.random() < mutate_prob:
                        self.weights_2[k,z,j] += (2*max_mutate_percent*np.random.random()-max_mutate_percent)

            for k in range(self.weights_3.shape[0]):
                for z in range(self.weights_3.shape[1]):
                    if np.random.random() < mutate_prob:
                        self.weights_3[k,z,j] += (2*max_mutate_percent*np.random.random()-max_mutate_percent)
            for k in range(self.bias_1.shape[0]):
                for z in range(self.bias_1.shape[1]):
                    self.bias_1[k,z,j] += (2*max_mutate_percent*np.random.random()-max_mutate_percent)
                    self.bias_2[k,z,j] += (2*max_mutate_percent*np.random.random()-max_mutate_percent)'''
        kernel = np.random.uniform(size=(self.weights_1.shape[0],self.weights_1.shape[1],taken[1]-taken[0]))
        kernel = kernel < mutate_prob
        self.weights_1[:,:,taken[0]:taken[1]][kernel] += (2*max_mutate_percent*np.random.uniform(size = (self.weights_1.shape[0],self.weights_1.shape[1],taken[1]-taken[0]))-max_mutate_percent)

        kernel = np.random.uniform(size=(self.weights_2.shape[0],self.weights_2.shape[1],taken[1]-taken[0]))
        kernel = kernel < mutate_prob
        self.weights_2[:,:,taken[0]:taken[1]][kernel] += (2*max_mutate_percent*np.random.uniform(size = (self.weights_2.shape[0],self.weights_2.shape[1],taken[1]-taken[0]))-max_mutate_percent)

        kernel = np.random.uniform(size=(self.weights_3.shape[0],self.weights_3.shape[1],taken[1]-taken[0]))
        kernel = kernel < mutate_prob
        self.weights_3[:,:,taken[0]:taken[1]][kernel] += (2*max_mutate_percent*np.random.uniform(size = (self.weights_3.shape[0],self.weights_3.shape[1],taken[1]-taken[0]))-max_mutate_percent)

        kernel = np.random.uniform(size=(self.bias_1.shape[0],self.bias_1.shape[1],taken[1]-taken[0]))
        kernel = kernel < mutate_prob
        self.bias_1[:,:,taken[0]:taken[1]][kernel] += (2*max_mutate_percent*np.random.uniform(size = (self.bias_1.shape[0],self.bias_1.shape[1],taken[1]-taken[0]))-max_mutate_percent)

        kernel = np.random.uniform(size=(self.bias_2.shape[0],self.bias_2.shape[1],taken[1]-taken[0]))
        kernel = kernel < mutate_prob
        self.bias_2[:,:,taken[0]:taken[1]][kernel] += (2*max_mutate_percent*np.random.uniform(size = (self.bias_2.shape[0],self.bias_2.shape[1],taken[1]-taken[0]))-max_mutate_percent)
    
    
    
    def generate_bottom_nn(self,percent_2):
        self.weights_1[:,:,int(percent_2*self.n_bots):self.n_bots] = 2*np.random.uniform(size=(self.weights_1.shape[0],self.weights_1.shape[1],self.weights_1.shape[2]-int(percent_2*self.weights_1)))-1
        self.weights_2[:,:,int(percent_2*self.n_bots):self.n_bots] = 2*np.random.uniform(size=(self.weights_2.shape[0],self.weights_2.shape[1],self.weights_2.shape[2]-int(percent_2*self.weights_2)))-1
        self.weights_3[:,:,int(percent_2*self.n_bots):self.n_bots] = 2*np.random.uniform(size=(self.weights_3.shape[0],self.weights_3.shape[1],self.weights_3.shape[2]-int(percent_2*self.weights_3)))-1
        self.bias_1[:,:,int(percent_2*self.n_bots):self.n_bots] = 2*np.random.uniform(size=(self.bias_1.shape[0],self.bias_1.shape[1],self.bias_1.shape[2]-int(percent_2*self.bias_1)))-1
        self.bias_2[:,:,int(percent_2*self.n_bots):self.n_bots] = 2*np.random.uniform(size=(self.bias_2.shape[0],self.bias_2.shape[1],self.bias_2.shape[2]-int(percent_2*self.bias_2)))-1
                                           






    def run(self,n_generations,drag_coeff,friction_coeff,inversion,g_force,gpositive,gnegative,min_height,min_drop_velocity,alpha,beta,n,percent_1 = 0.1,percent_2 = 0.5,mutate_prob = 0.4,verbose=True):
        for i in range(n_generations):
            for j in range(self.n_bots):
                delta_energy = 0
                for k in range(self.n_tracksegments):
                    track_angle = self.compute_track_angles(i,j,k)
                    delta_x,delta_y = self.compute_x_y_nodes(i,j,k,track_angle)
                    self.compute_generation_outputs(i,j,k,track_angle,delta_y,delta_energy)
                    self.update_inputs(i,j,k)
                    radius = self.compute_segment_radius(i,j,k)
                    delta_energy = self.compute_energy_loss(i,j,k,radius,drag_coeff,friction_coeff)
            for j in range(self.n_bots):
                self.y_splines = self.y_nodes
                for k in range(self.n_tracksegments):
                    self.y_splines_deri_1[k,j] = 0.5*np.tan(self.track_angles[k+1,j])+np.tan(self.track_angles[k,j])
                self.y_splines_deri_1[self.n_tracksegments,j]=np.tan(self.track_angles[k,j])
                self.compute_roc(i,j)
                self.compute_velocities(i,j)
                self.centripetal_acc = (self.velocities^2)/(self.g*self.roc)
                self.compute_gforce(i,j)
            for j in range(self.n_bots):
                min_velocity = 0
                for k in range(self.n_tracksegments):
                    self.penalty_for_moving_sideways(i,j,k,inversion)
                    self.update_inversion_score(i,j,k,inversion,g_force)
                    self.penalty_for_height_voilation(i,j,k,min_height,inversion)
                    self.update_gforce_score(i,j,k,g_force,gpositive,gnegative,inversion)
                    self.update_velocity_score(i,j,k,inversion,alpha)
                    self.penalty_for_velocity_voilation(i,j,k,min_drop_velocity,inversion)
                    self.update_steepness_score(i,j,k,beta)
            sorted_score , sorted_ind = self.sort_normalize_score(n,verbose)
            self.store_nn_for_top_bots(sorted_ind,percent_1)
            self.mutate_nn(percent_1,percent_2,mutate_prob)
            self.generate_bottom_nn(percent_2)



