import numpy as np
import uaibot as ub
from create_franka_emika_3_mod import *
import matplotlib.pyplot as plt
robot = create_franka_emika_3_mod()
# matplotlib.use('Qt5Agg') 




# #Obstacles
# obstacles = []
# obstacles.append(ub.Box(htm = ub.Utils.trn([0.53, 0.16, 0.45]), width=0.35,depth=0.05,height=0.90,color='magenta'))
# obstacles.append(ub.Box(htm = ub.Utils.trn([0.53,-0.16, 0.45]), width=0.35,depth=0.05,height=0.90,color='magenta'))
# obstacles.append(ub.Box(htm = ub.Utils.trn([0.53, 0.00, 0.925]), width=0.35,depth=0.35,height=0.05,color='magenta'))


#Obstacles
obstacles = []
obstacles.append(ub.Box(htm = ub.Utils.trn([0.52, 0.19, 0.48]), width=0.32,depth=0.08,height=0.92,color='magenta'))
obstacles.append(ub.Box(htm = ub.Utils.trn([0.52,-0.19, 0.48]), width=0.32,depth=0.08,height=0.92,color='magenta'))
obstacles.append(ub.Box(htm = ub.Utils.trn([0.52, 0.00, 0.96]), width=0.32,depth=0.46,height=0.08,color='magenta'))



#Initial configuration
q = np.matrix([[ 1.0582, -1.3811,  0.3629, -1.9647, -0.959,   1.4881, -0.1534]]).T
#Target pose
htm_tg = ub.Utils.trn([0.64,0,0.75])*ub.Utils.roty(np.pi/2)*ub.Utils.rotz(np.pi/2)

#Control parameters
dt = 0.01
K = 8*np.diag([0.2,0.2,0.2,0.02,0.02,0.02])
K = 2*np.diag([0.2,0.2,0.2,0.2,0.2,0.2])
eta = 0.2*0+0.5

#mode = 0 (Euclidean)
#mode = 1 (Smoothed)
mode = 1

if mode == 0:
    h=1e-6
    eps=0
    delta=0.03
else:
    h=0.1
    eps=0.01
    delta=0.002

###################################################################


sim = ub.Simulation(background_color='lightblue')
sim.add(robot)
robot.add_ani_frame(0,q)

for obs in obstacles:
    sim.add(obs)
    
    

frame_tg = ub.Frame(htm=htm_tg)
sim.add(frame_tg)

ball1 = ub.Ball(color='yellow', radius=0.015)
ball2 = ub.Ball(color='cyan', radius=0.015)
sim.add([ball1, ball2])
###################



q = np.matrix(robot.q)

hist_u = []
for i in range(3500):
    
    mat_A = np.matrix(np.zeros((0,7)))
    mat_b = np.matrix(np.zeros((0,1)))
    
    true_dist = 1000
    for obs in obstacles:
        dr = robot.compute_dist(q=q, obj=obs, h=h, eps=eps)
        mat_A = np.vstack((mat_A, dr.jac_dist_mat))
        mat_b = np.vstack((mat_b, -eta*(dr.dist_vect-delta)))
        
        true_dist = min(true_dist, robot.compute_dist(q=q,obj=obs).get_closest_item().distance)
        
    mat_A = np.vstack((mat_A, np.identity(7)))
    mat_b = np.vstack((mat_b, -eta*(q-robot.joint_limit[:,0])))
    
    mat_A = np.vstack((mat_A,-np.identity(7)))
    mat_b = np.vstack((mat_b,-eta*(robot.joint_limit[:,1]-q)))
    
    r, jac_r = robot.task_function(q=q, htm_tg=htm_tg)
    
    mat_H = jac_r.T*jac_r + 0.01*np.identity(7)
    mat_f = jac_r.T*(K*r)
    
    u = ub.Utils.solve_qp(mat_H, mat_f, mat_A, mat_b)
    u_np = np.array(u)
    
    hist_u.append(np.squeeze(u_np, axis=1))
    
    min_d = 1000
    for obs in obstacles:
        ci = robot.compute_dist(q=q, obj=obs).get_closest_item()
        if ci.distance < min_d:
            min_d = ci.distance
            point_link = ci.point_link
            point_object = ci.point_object
            
    ball1.add_ani_frame(i*dt,ub.Utils.trn(point_link))
    ball2.add_ani_frame(i*dt,ub.Utils.trn(point_object))

    q+=u*dt
    
    print("D = "+str(round(true_dist,3))+", "+str(r.T))
    
    robot.add_ani_frame(i*dt,q)
    


sim.save(file_name="test")

hist_u = np.array(hist_u)
plt.plot(hist_u)
plt.show()
if mode == 0:
    plt.savefig("euclidean.png")
else:
    plt.savefig("smoothed.png")
#sim.run("test")