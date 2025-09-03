import uaibot as ub
import numpy as np
import sys
import time
import matplotlib.pyplot as plt

def create_pyramid(htm, radius, height):
    """
    Create a 4-sided pyramid using Uaibot's ConvexPolytope.
    The square base lies in the plane z=0, centered at (0,0,0),
    with side length 2*radius, and the apex is at (0,0,height).

    :param htm:    4x4 homogeneous transform giving the final pose
    :param radius: half of the base's side length
    :param height: height of the pyramid (apex at z = height)
    :return:       a uaibot ConvexPolytope object
    """

    # Define the apex in local coordinates.
    apex = np.array([0, 0, height])

    # We'll collect A and b for the half-spaces A*x <= b.
    A_list = []
    b_list = []

    # Number of sides in the base (4 -> a square).
    no_sides = 4

    # Loop over each side of the square base
    for i in range(no_sides):
        # Corner i of the base
        theta1 = 2 * np.pi * i / no_sides
        # Corner i+1 (the "next" corner)
        theta2 = 2 * np.pi * (i + 1) / no_sides

        # Points in R^3 (local frame)
        p1 = apex  # apex
        p2 = np.array([radius * np.cos(theta1), radius * np.sin(theta1), 0])
        p3 = np.array([radius * np.cos(theta2), radius * np.sin(theta2), 0])

        # Normal vector to the plane (p1, p2, p3)
        normal = np.cross(p2 - p1, p3 - p1)

        # We want the "inside" of the plane to include the center of the base (0,0,0).
        # So we check the sign of normal dot (center - apex).
        center = np.array([0, 0, 0])
        if np.dot(normal, center - p1) > 0:
            normal = -normal

        # The plane is normal^T x <= normal^T p1
        A_list.append(normal)
        b_list.append(np.dot(normal, p1))

    # Finally, add the plane for z >= 0, i.e. -z <= 0
    A_list.append([0, 0, -1])
    b_list.append(0)

    # Convert everything to numpy arrays
    A = np.array(A_list, dtype=float)
    b = np.array(b_list, dtype=float).reshape(-1, 1)

    # Build the convex polytope at the local frame
    pyramid = ub.ConvexPolytope(A=A, b=b, htm=ub.Utils.trn([0, 0, 0])*ub.Utils.rotz(np.pi/4), color="yellow")

    # Now apply the desired pose 'htm'
    pyramid.add_ani_frame(time=0, htm=htm)

    return pyramid
   
 
def funF(r):
#Function that is used in the control loop

    f = np.matrix(r)
    for j in range(np.shape(r)[0]):
        f[j,0] = np.sign(r[j,0])*np.sqrt(np.abs(r[j,0]))
        
    return f

##############################################################################
#Create list of obstacles

# cone = create_pyramid(ub.Utils.trn([0.42,0,0.16]),radius=0.26/np.sqrt(2),height=0.45)
# table_top = ub.Box(ub.Utils.trn([0.4,0,0.08]),width=0.32,depth=0.48,height=0.16, color='magenta')


cone = create_pyramid(ub.Utils.trn([0.42,0,0.44])*ub.Utils.rotz(-np.pi/4),radius=0.26/np.sqrt(2),height=0.45)
table_top = ub.Box(ub.Utils.trn([0.4,0,0.22]),width=0.32,depth=0.48,height=0.44, color='magenta')

obstacles = [cone, table_top]

##############################################################################
#Create the robot
robot = ub.Robot.create_franka_emika_3()
no_joint = np.shape(robot.q)[0]

##############################################################################
#Create the initial pose and target pose

# htm_init = ub.Utils.trn([0.20, 0.0, 0.55]) * ub.Utils.rotx(np.pi)
# htm_des = ub.Utils.trn([0.60, 0.0, 0.57]) * ub.Utils.rotx(np.pi)
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
# q_init = np.matrix([[0.00],[-0.965],[0.00],[-2.45],[0.00 ],[ 1.50],[ 0.79]])
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
sim.save(".", "exp3")
    
