from utils_paper import *
import uaibot as ub
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
from create_franka_emika_3_mod import *


##############################################################################
#Create list of obstacles
# cone = create_cone(ub.Utils.trn([0.45,0,0.5]),radius=0.1,height=0.5)
# table_top = ub.Box(ub.Utils.trn([0.5,0,0.4]),width=0.6,depth=0.8,height=0.2, color='magenta')

obs_box = ub.Box(ub.Utils.trn([0.46,0,0.44+0.23])*ub.Utils.rotz(0*np.pi/4),width=0.16,depth=0.16,height=0.46, color='yellow')
table_top = ub.Box(ub.Utils.trn([0.4,0,0.22]),width=0.32,depth=0.48,height=0.44, color='magenta')

obstacles = [obs_box, table_top]

##############################################################################
#Create the robot
#Usually we would create using something like robot = ub.Robot.create_franka_emika_3()
#But the franka emika model used in the experiments has slightly different parameters than the one in official UAIBot
#package, so I created a custom 'create_franka_emika_3_mod' function

robot = create_franka_emika_3_mod()
no_joint = np.shape(robot.q)[0]

##############################################################################
#Create the initial pose and target pose

# htm_init = ub.Utils.trn([0.45, -0.3, 0.7])*ub.Utils.roty(np.pi/2)
htm_init = ub.Utils.trn([0.45, -0.3, 0.7])*ub.Utils.roty(-np.pi/2)*ub.Utils.rotx(np.pi)
htm_des = ub.Utils.trn([0.40, 0.16, 0.7])*ub.Utils.roty(np.pi/2)*ub.Utils.rotx(np.pi/6)*ub.Utils.rotz(np.pi/2)

##############################################################################
#Find an initial configuration that is non colliding
#Only need to run it once, then you can hard-code the initial configuration

# cont = True

# while cont:
#     q_init = robot.ikm(htm_init)
#     isfree, message, info = robot.check_free_configuration(q=q_init, obstacles=obstacles)
#     cont = not isfree
    
#Add initial config to the robot
q_init = np.matrix([[ 0.8004],[-0.9493],[-1.0221],[-2.1716],[-0.309 ],[ 3.1311],[ 0.27]])

robot.add_ani_frame(time=0,q=q_init)

##############################################################################
#Create the simulation environment. This is just necessary to run the animation, no need for this in the real
#robot

sim = ub.Simulation(background_color='#191919')

#Add the obstacles
sim.add(obstacles)
#Add the robot
sim.add(robot)
#Add the two frames (starting and target) as visual objects
sim.add(ub.Frame(htm_init,size=0.1))
sim.add(ub.Frame(htm_des,size=0.1))

##############################################################################
#Main control loop


#Parameters
dt = 0.005 #0.01
#CBF eta
eta = 0.8
#Convergence gain
kconv = 0.5
#Smoothing parameter eps for obstacle avoidance
eps_to_obs = 0.05
#Smoothing parameter h for obstacle avoidance
h_to_obs = 0.1
#Smoothing parameter eps for auto collision avoidance
eps_auto = 0.02
#Smoothing parameter h for auto collision avoidance
h_auto = 0.05
#Safety margin obstacle avoidance
d_safe_obs = 0.005
#Safety margin auto collision avoidance
d_safe_auto = 0.002
#Safety margin joint
d_safe_jl = (np.pi/180)*1
#Error tolerance position (in mm)
tolp=1.0 
#Error tolerance angle (in def)
tolo=1.0 
#Regularization term for the controller
eps_reg = 0.01
#Maximum time
Tmax = 15


#UNCOMMENT THE NEXT LINES TO RUN IN THE EUCLIDEAN DISTANCE MODE####
eps_to_obs = 0
h_to_obs = 0
d_safe_obs = 0.05

eps_auto = 0
h_auto = 0
d_safe_auto = 0.02
####################################################################


hist_time = [] 

hist_dist_0 = []
hist_dist_sm = []

hist_q_dot = []

p_init = np.matrix([0,0,0]).transpose()

for i in range(round(Tmax/dt)):
    t = i*dt
    
    #Get current configuration, in the real robot you have to measure this from somewhere
    q = np.matrix(robot.q)
    
    start = time.time()

    #Compute task function and Jacobian at current configuration
    r, jac_r = robot.task_function(q=q, htm_des = htm_des)
    
    #GATHER DATA FOR THE OBSTACLE AVOIDANCE CONSTRAINT
    #Loop all the obstacles and compute distance info
    #Collect the matrices A_obs and b_obs:
    # A_obs: each line has the smooth metric gradient (in configuration space), as a row matrix,
    #        from an object of the robot and an object of the obstacle
    # b_obs: each line has the smooth metric
    A_obs = np.matrix(np.zeros((0,no_joint)))
    b_obs = np.matrix(np.zeros((0,1)))
    
    for obs in obstacles:
        dist_info = robot.compute_dist(obj = obs, q=q, h=h_to_obs, eps = eps_to_obs)

        A_obs = np.vstack((A_obs, dist_info.jac_dist_mat))
        b_obs = np.vstack((b_obs, dist_info.dist_vect))
        
    #GATHER DATA FOR THE JOINT LIMIT AVOIDANCE COINSTRANT
    q_min = robot.joint_limit[:,0]
    q_max = robot.joint_limit[:,1]
    A_joint = np.matrix(np.vstack(  (np.identity(7), -np.identity(7))  ))
    b_joint = np.matrix(np.vstack(  (q-q_min , q_max - q)  ))
    
    #GATHER DATA FOR THE AUTO COLLISION AVOIDANCE CONSTRAINT
    dist_info =robot.compute_dist_auto(q = q, eps=eps_auto, h=h_auto)
    #We can safely ignore the last four collisions, that is why we have the [0:-4,:]   
    A_auto = dist_info.jac_dist_mat[0:-4,:]
    b_auto = dist_info.dist_vect[0:-4,:]    
    
    #CREATE THE QUADRATIC PROGRAM    
    H_mat = jac_r.transpose()*jac_r + eps_reg*np.identity(no_joint)
    f_mat = kconv * jac_r.transpose()*funF(r)
    
    A_mat = np.vstack((A_obs,A_joint,A_auto))
    b_mat = np.vstack( (b_obs-d_safe_obs, b_joint-d_safe_jl, b_auto-d_safe_auto) )
    
    b_mat = -eta*b_mat
        
    #Compute qdot
    qdot = ub.Utils.solve_qp(H_mat, f_mat, A_mat, b_mat)
    

    end = time.time()
    hist_time.append(end - start)

    #Here, you send the velocity command do the robot
    #Since this is a simulation, we make a first order integration and add the animation
    #frame to the simulation
    q_next = q + qdot*dt
    robot.add_ani_frame(time=t,q=q_next)
    
    #Print some statitstics
    
    #Compute some statics
    epx = abs(round(1000*r[0,0]))
    epy = abs(round(1000*r[1,0]))
    epz = abs(round(1000*r[2,0]))
    eox = round((180/np.pi)*np.arccos(1-r[3,0]),2)
    eoy = round((180/np.pi)*np.arccos(1-r[4,0]),2)
    eoz = round((180/np.pi)*np.arccos(1-r[5,0]),2)
    
    
    str_msg = "\rt = "+str(round(t,2))+" s, epx = "+str(epx)+" mm, epy = "+str(epy)+" mm, epz = "+str(epz)+" mm, "
    str_msg += "eox = "+str(epx)+" deg, eoy = "+str(epy)+" deg, eoz = "+str(epz)+" deg     "


    sys.stdout.write(str_msg)
    sys.stdout.flush()
    
#     robot.update_col_object(time=t)
#     col_obj = robot.links[-1].col_objects[-1][0]
#     p_init, _, ds, _ = col_obj.compute_dist(obj = cone, h=h_to_obs,eps=eps_to_obs, p_init=p_init, no_iter_max=1000)
#     _, _, d0, _ = col_obj.compute_dist(obj = cone)
    
    hist_q_dot.append(qdot)
    

    
#     # hist_dist_0.append(np.linalg.norm(robot.compute_dist(obj = obs, q=q, h=0, eps = 0).get_item(6,1).jac_distance))
#     # hist_dist_sm.append(np.linalg.norm(robot.compute_dist(obj = obs, q=q, h=h_to_obs, eps = eps_to_obs).get_item(6,1).jac_distance))
    
# def linear_regression(x, y):
#     n = len(x)
#     if n != len(y):
#         raise ValueError("x and y must have the same length")

#     mean_x = sum(x) / n
#     mean_y = sum(y) / n

#     numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
#     denominator = sum((xi - mean_x) ** 2 for xi in x)

#     if denominator == 0:
#         raise ValueError("Denominator is zero; all x values might be the same")

#     a = numerator / denominator
#     b = mean_y - a * mean_x

#     return a, b

# # a, b = linear_regression(hist_dist_sm, hist_dist_0)  
# # plt.plot([a*x+b for x in hist_dist_sm])

# # plt.plot(hist_dist_sm)
# # plt.plot(hist_dist_0)

q_dot_l = [ [v[0] for v in u.tolist()] for u in hist_q_dot]

for i in range(7):
    plt.plot([u[i] for u in q_dot_l])
    plt.show()


# # plt.plot(hist_time)
# # plt.show()

# #Put here the path to the folder where you want to save the 3D animation of the controller.
# #Then, just go there and open using any browser
sim.save(".", "exp4")
    
