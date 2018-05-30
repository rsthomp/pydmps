import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import metrics
from keras import regularizers
from sklearn import preprocessing
import pydmps.dmp_discrete

#set coupling term in x direction as 0
#Exponentially reduce coupling term to 0 on passing
#Set coupling term to 0 if obstacle is beyond the goal

#vector between obstacle center adn end effector (2)
#motion duration multiplied velocity of end effector
#distance to obstacle
#angle between end effector velocity and obstacle

#labels will be P


beta = 20.0 / np.pi
gamma = 300
R_halfpi = np.array([[np.cos(np.pi / 2.0), -np.sin(np.pi / 2.0)],
                     [np.sin(np.pi / 2.0), np.cos(np.pi / 2.0)]])
num_obstacles = 10
obstacles = [[-.5, .02] for i in range(10)]#np.random.random((num_obstacles, 2))* 2 - 1

global c_t
global obst_vec
global obst_dist
global cur_vel
global obst_angle

dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=2, n_bfs=10,
                                        w=np.zeros((2,10)), dt=.01)

c_t = np.zeros((dmp.timesteps * num_obstacles, dmp.n_dmps))
obst_vec = np.zeros((dmp.timesteps * num_obstacles, dmp.n_dmps))
obst_dist = np.zeros((dmp.timesteps * num_obstacles, dmp.n_dmps))
cur_vel = np.zeros(dmp.timesteps * num_obstacles)
obst_angle = np.zeros(dmp.timesteps* num_obstacles)

def avoid_obstacles(y, dy, goal, t):
    global c_t
    global obst_vec
    global obst_dist
    global cur_vel
    global obst_angle
    p = np.zeros(2)

    for obstacle in obstacles:
        # based on (Hoffmann, 2009)

        # if we're moving
        if np.linalg.norm(dy) > 1e-5:
            
            # get the angle we're heading in
            phi_dy = -np.arctan2(dy[1], dy[0])
            obst_dist[t] = dy
            R_dy = np.array([[np.cos(phi_dy), -np.sin(phi_dy)],
                             [np.sin(phi_dy), np.cos(phi_dy)]])
            # calculate vector to object relative to body
            obj_vec = obstacle - y
            obst_vec[t] = obj_vec

            # rotate it by the direction we're going
            obj_vec = np.dot(R_dy, obj_vec)
            # calculate the angle of obj relative to the direction we're going
            phi = np.arctan2(obj_vec[1], obj_vec[0])
            obst_angle[t] = phi

            dphi = phi * np.exp(-beta * abs(phi)) #* gamma
            R = np.dot(R_halfpi, np.outer(obstacle - y, dy))
            pval = -np.nan_to_num(np.dot(R, dy) * dphi)
            c_t[t] = pval
            cur_vel[t] = np.linalg.norm(obj_vec)

            # check to see if the distance to the obstacle is further than
            # the distance to the target, if it is, ignore the obstacle
            if np.linalg.norm(obj_vec) > np.linalg.norm(goal - y):
                pval = 0

            p += pval * gamma
        t += 100
    return p

# test normal run


ty_track = np.zeros((dmp.timesteps, dmp.n_dmps))
tdy_track = np.zeros((dmp.timesteps, dmp.n_dmps))
tddy_track = np.zeros((dmp.timesteps, dmp.n_dmps))


goals = [[-1, 0] for i in range(10)]#[[np.cos(theta), np.sin(theta)] for theta in np.linspace(0, 2*np.pi, 20)[:-1]]

for goal in goals:
    dmp.goal = goal
    dmp.reset_state()
    for t in range(dmp.timesteps):
        ty_track[t], tdy_track[t], tddy_track[t] = dmp.step(external_force=avoid_obstacles(dmp.y, dmp.dy, goal, t))

model = Sequential()
model.add(Dense(10, activation='relu', bias_regularizer=regularizers.l1(0.01), input_dim=3))
model.add(Dense(2, activation='tanh'))



inpt = np.vstack((obst_vec[:,0], obst_vec[:,1], cur_vel))
inpt = np.swapaxes(inpt, 0, 1)
inpt = preprocessing.scale(inpt)

model.compile(optimizer='adagrad',
              loss='mse', metrics=[metrics.mse])

label = c_t



model.fit(inpt, label, epochs=100, batch_size=10, verbose=2)

y_track = np.zeros((dmp.timesteps, dmp.n_dmps))
dy_track = np.zeros((dmp.timesteps, dmp.n_dmps))
ddy_track = np.zeros((dmp.timesteps, dmp.n_dmps))

""" return a p, the (Cx, Cy) coupling term 
to avoid an obstacle"""
def nn_avoid(y, dy, goal, model, postprocess=True):
    p = np.zeros(2)
    for obstacle in obstacles:
        if np.linalg.norm(dy) > 1e-5:
                
                # get the angle we're heading in
                phi_dy = -np.arctan2(dy[1], dy[0])
                obstd = dy

                R_dy = np.array([[np.cos(phi_dy), -np.sin(phi_dy)],[np.sin(phi_dy), np.cos(phi_dy)]])
               
                # calculate vector to object relative to body
                obj_vec = obstacle - y
                obstv = obj_vec

                # rotate it by the direction we're going
                obj_vec = np.dot(R_dy, obj_vec)
                # calculate the angle of obj relative to the direction we're going
                phi = np.arctan2(obj_vec[1], obj_vec[0])
                obsta = phi
                cv = np.linalg.norm(obj_vec)
                tst = np.array([[obstv[0]], [obstv[1]], [cv]])#, [obstd[0]], [obstd[1]], [obsta], [cv]])
                tst = np.swapaxes(tst, 0, 1)
                #print(tst)

                pval = model.predict(tst, batch_size=1)
                pval = np.array([pval[0][0], pval[0][1]])
                
                p += pval * gamma
        if postprocess:
            return postprocess_output(p, y, obstacles[0], goal)
        return p


def get_straight_path(start, goal, timesteps, n_dmps, oscillate = False):
    #constantly make progress to the goal, always move at that angle. 
    dmp_paths = []
    for i in range(n_dmps):
        dmp_paths.append(np.linspace(start[i], goal[i], timesteps))
    if oscillate:
        A = .05
        dmp_paths[1] = [dmp_paths[1][t] + A*np.sin(30*t) for t in range(len(dmp_paths[0]))]
    path = np.vstack(dmp_paths)
    return path

""" postprocessing steps to ensure safe behaviour"""
def postprocess_output(x, state, obstacle, goal):
    Cy_original = x[1] #ignore x[0] (the paper told me to do it!!)
    Cy = Cy_original
    #exponentially reduce coupling term to 0 on passing the obstacle
    k = 100
    if abs(state[0]) > abs(obstacle[0]):
        Cy = Cy_original*np.e**(-k*(obstacle[0]-state[0])**2)
    #set coupling term to 0 if obstacle is beyond the goal
    if abs(state[0]) > abs(goal[0]):
        Cy = 0
    return [0, Cy]


path = get_straight_path([0,0], [-1,0.015], dmp.timesteps, dmp.n_dmps, oscillate=False)
dmp.imitate_path(y_des = path, plot = False)

for goal in goals:
    dmp.goal = goal
    dmp.reset_state()
    for t in range(dmp.timesteps):
        #print("NN: " + str(nn_avoid(dmp.y, dmp.dy, goal, model)))
        #print("AO: " + str(avoid_obstacles(dmp.y, dmp.dy, goal, t)))
        y_track[t], dy_track[t], ddy_track[t] = dmp.step(external_force=nn_avoid(dmp.y, dmp.dy, goal, model, postprocess=True))



    plt.figure(1, figsize=(6,6))
    plot_goal, = plt.plot(dmp.goal[0], dmp.goal[1], 'gx', mew=3)
    for obstacle in obstacles:
        plot_obs, = plt.plot(obstacle[0], obstacle[1], 'rx', mew=3)
    plot_path, = plt.plot(y_track[:,0], y_track[:, 1], 'b', lw=2)
    plot_path, = plt.plot(ty_track[:,0], ty_track[:, 1], 'g', lw=2)
    plot_path, = plt.plot(path[:,0], path[:, 1], 'y', lw=2)
    plt.title('DMP system - obstacle avoidance')

plt.axis('equal')
plt.xlim([-1.1,1.1])
plt.ylim([-1.1,1.1])
plt.show()


            


