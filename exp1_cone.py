import uaibot as ub
import numpy as np
import sys
import time
import matplotlib.pyplot as plt

def create_cone(htm, radius, height):
#Create a cone-like object using Uaibot's capabilities of creating convex polytopes 
#defined by Ap<=b. 'htm' is the homogeneous transformation matrix representing the pose of the cylinder.
#'radius' is the radius of the base of the cone, and 'height' the height of the cone

    #Create the side inequalities Ap<=b for the cone
    no_sides = 6
    A = np.matrix(np.zeros((0,3)))
    b = np.matrix(np.zeros((0,1)))


    phi = np.arcsin(radius/np.sqrt(height**2+radius**2))

    for i in range(no_sides):
        theta = 2*np.pi*i/no_sides
        a = np.matrix([np.cos(phi)*np.cos(theta), np.cos(phi)*np.sin(theta), np.sin(phi)])
        A = np.vstack((A,a))
        b = np.vstack((b,radius))
        
    ##Add an inequality in the bottom (to cover the cone from the bottom)
    A = np.vstack((A,np.matrix([0,0,-1])))
    b = np.vstack((b,0))
    
    #Create the convex polytope
    cone = ub.ConvexPolytope(A=A,b=b,htm=ub.Utils.trn([0,0,0]),color='yellow')
    
    #Set the desired htm at time t=0
    cone.add_ani_frame(time=0,htm=htm)
    
    return cone
   
 
def funF(r):
#Function that is used in the control loop

    f = np.matrix(r)
    for j in range(np.shape(r)[0]):
        f[j,0] = np.sign(r[j,0])*np.sqrt(np.abs(r[j,0]))
        
    return f

##############################################################################
#Create list of obstacles
# cone = create_cone(ub.Utils.trn([0.45,0,0.5]),radius=0.1,height=0.5)
# table_top = ub.Box(ub.Utils.trn([0.5,0,0.4]),width=0.6,depth=0.8,height=0.2, color='magenta')

cone = create_cone(ub.Utils.trn([0.44,0,0.44]),radius=0.11,height=0.45)
table_top = ub.Box(ub.Utils.trn([0.4,0,0.22]),width=0.32,depth=0.48,height=0.44, color='magenta')

obstacles = [cone, table_top]

##############################################################################
#Create the robot
robot = ub.Robot.create_franka_emika_3()
no_joint = np.shape(robot.q)[0]

##############################################################################
#Create the initial pose and target pose

# htm_init = ub.Utils.trn([0.45, -0.3, 0.7])*ub.Utils.roty(np.pi/2)
htm_init = ub.Utils.trn([0.45, -0.3, 0.7])*ub.Utils.roty(-np.pi/2)*ub.Utils.rotx(np.pi)
htm_des = ub.Utils.trn([0.45, 0.16, 0.7])*ub.Utils.roty(np.pi/2)*ub.Utils.rotx(np.pi/6)*ub.Utils.rotz(np.pi/2)

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
dt = 0.01
#CBF eta
eta = 0.8
#Convergence gain
kconv = 1.5
#Smoothing parameter eps for obstacle avoidance
eps_to_obs = 0.003
#Smoothing parameter h for obstacle avoidance
h_to_obs = 0.003
#Smoothing parameter eps for auto collision avoidance
eps_auto = 0.02
#Smoothing parameter h for auto collision avoidance
h_auto = 0.05
#Safety margin obstacle avoidance
d_safe_obs = 0.016
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

hist_time = [] 

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




# plt.plot(hist_time)
# plt.show()

#Put here the path to the folder where you want to save the 3D animation of the controller.
#Then, just go there and open using any browser
sim.save(".", "exp1")
    
