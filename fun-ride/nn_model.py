#from .model import NN_Model
from .nn_parameters import n_inner_neurons,n_bots,n_generations
from .coaster_parameters import intial_velocity,max_height ,intial_height,intial_track_angle,track_lenght,tracksegment_lenght,drag_coef,friction_coef,min_height,min_drop_velocity

inversion = True
g_force = (0,4)
gpositive=0.3
gnegative = 0.0

alpha = 0.1
beta = 0.1

#a = NN_Model(n_bots,track_lenght,tracksegment_lenght,n_inner_neurons,intial_velocity,max_height,intial_height, intial_track_angle)
#a.run(n_generations,drag_coef,friction_coef,inversion,g_force,gpositive,gnegative,min_height,min_drop_velocity,alpha,beta,n,percent_1 = 0.1,percent_2 = 0.5,mutate_prob = 0.4,plot_hilloff = True,verbose=True)