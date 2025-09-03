import uaibot as ub
import numpy as np


def funF(r):
#Function that is used in the control loop

    f = np.matrix(r)
    for j in range(np.shape(r)[0]):
        f[j,0] = np.sign(r[j,0])*np.sqrt(np.abs(r[j,0]))
        
    return f

robot = ub.Robot.create_franka_emika_3()
q_init = np.array([0.0, -0.86, 0.0, -1.98, 0.0, 2.68, 0.78])
robot.add_ani_frame(time=0,q=q_init)
sim = ub.Simulation([robot])

table_top = ub.Box(ub.Utils.trn([0.4,0,0.22]),width=0.32,depth=0.48,height=0.44, color='magenta')

obstacles = [table_top]
sim.add(obstacles)

# Define the time-varying trajectory
xc = 0.45
yc = 0
zc = 0.70
radius = 0.20
period = 10.0 #in seconcs
phase = np.pi/2

T_des = lambda t: ub.Utils.trn([xc, yc+radius*np.cos(2*np.pi*t/period+phase), zc+radius*np.sin(2*np.pi*t/period+phase)])*ub.Utils.roty(np.pi/2)*ub.Utils.rotz(np.pi)

#Create a point cloud to show the position part of the trajectory
pc_data = [T_des(i*period/1000)[0:3,-1] for i in range(1000)]
pc = ub.PointCloud(points = pc_data, color='orange', size=0.01)
sim.add(pc)

#Control parameters
dt=0.01
K = 1.5

#Start the main loop
N = round(3*period/dt)
q = np.matrix(robot.q)

joint_lb = np.array([-2.3093, -1.5133, -2.4937, -2.7478, -2.48, 0.8521, -2.6895])
joint_ub = np.array([2.3093, 1.5133, 2.4937, -0.4461, 2.48, 4.2094, 2.6895])

q_mid = ((joint_ub+joint_lb)/2).reshape(-1,1)
Kp_joint = np.diag([1, 1, 1, 1, 4, 2, 1])*1.0

no_joint = np.shape(robot.q)[0]

#CBF eta
eta = 2.0
#Convergence gain
K = 1.0
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
#Regularization term for the controller
eps_reg = 0.01
#Encourage to center
eps_centering = 0.01

for i in range(N):
    t = i*dt
    #Compute the task function and its Jacobian
    r , jac_r = robot.task_function(q=q, htm_des=T_des(t))

    #Compute the task feedforward numerically: partial r(q,t)/partial t
    r_next , _ = robot.task_function(q=q, htm_des=T_des(t+dt))
    r_prev , _ = robot.task_function(q=q, htm_des=T_des(t-dt))
    ff = (r_next-r_prev)/(2*dt)
    W = np.diag(1.0/(joint_ub-joint_lb))

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
    H_mat = jac_r.transpose()*jac_r + eps_centering * W @ W
    f_mat = jac_r.transpose() * (K*r+ff) - eps_centering * W @ q_mid
    
    A_mat = np.vstack((A_obs,A_joint,A_auto))
    b_mat = np.vstack( (b_obs-d_safe_obs, b_joint-d_safe_jl, b_auto-d_safe_auto) )
    
    b_mat = -eta*b_mat
        
    #Compute qdot
    qdot = ub.Utils.solve_qp(H_mat, f_mat, A_mat, b_mat)

    #Send to the robot
    q += qdot*dt
    robot.add_ani_frame(time=t,q=q)
    
    print(r.transpose())



sim.save(".","exp2")