'''
Copyright (C) 2016 Travis DeWolf

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import numpy as np
import matplotlib.pyplot as plt
import pdb
import pydmps.dmp_discrete

beta = 20.0 / np.pi
gamma = 100
R_halfpi = np.array([[np.cos(np.pi / 2.0), -np.sin(np.pi / 2.0)],
                     [np.sin(np.pi / 2.0), np.cos(np.pi / 2.0)]])
num_obstacles = 1
obstacles = [[0.5,0.01]]

def avoid_obstacles(y, dy, goal):
    p = np.zeros(2)

    for obstacle in obstacles:
        # based on (Hoffmann, 2009)

        # if we're moving
        if np.linalg.norm(dy) > 1e-5:

            # get the angle we're heading in
            phi_dy = -np.arctan2(dy[1], dy[0])
            R_dy = np.array([[np.cos(phi_dy), -np.sin(phi_dy)],
                             [np.sin(phi_dy), np.cos(phi_dy)]])
            # calculate vector to object relative to body
            obj_vec = obstacle - y
            # rotate it by the direction we're going
            obj_vec = np.dot(R_dy, obj_vec)
            # calculate the angle of obj relative to the direction we're going
            phi = np.arctan2(obj_vec[1], obj_vec[0])

            dphi = gamma * phi * np.exp(-beta * abs(phi))
            R = np.dot(R_halfpi, np.outer(obstacle - y, dy))
            pval = -np.nan_to_num(np.dot(R, dy) * dphi)

            # check to see if the distance to the obstacle is further than
            # the distance to the target, if it is, ignore the obstacle
            if np.linalg.norm(obj_vec) > np.linalg.norm(goal - y):
                pval = 0

            p += pval
    return p

def get_straight_path(start, goal, timesteps, n_dmps, oscillate = False):
    #constantly make progress to the goal, always move at that angle. 
    dmp_paths = []
    for i in range(n_dmps):
        dmp_paths.append(np.linspace(start[i], goal[i], timesteps))
    if oscillate:
        A = 20
        dmp_paths[1] = [dmp_paths[1][t] + A*np.sin(0.01*t) for t in range(len(dmp_paths[0]))]
    path = np.vstack(dmp_paths)
    return path

# test normal run
dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=2, n_bfs=10,
                                        w=np.zeros((2,10)))
y_track = np.zeros((dmp.timesteps, dmp.n_dmps))
dy_track = np.zeros((dmp.timesteps, dmp.n_dmps))
ddy_track = np.zeros((dmp.timesteps, dmp.n_dmps))
goals = [[1.0, 0.015]] #just one for now
goal = goals[0]
path = get_straight_path([0,0], goal, dmp.timesteps, dmp.n_dmps, oscillate=True)
dmp.imitate_path(y_des = path, plot = False)

for goal in goals:
    dmp.goal = goal
    dmp.reset_state()
    for t in range(dmp.timesteps):
        y_track[t], dy_track[t], ddy_track[t] = \
                dmp.step()

    plt.figure(1, figsize=(6,6))
    plot_goal, = plt.plot(dmp.goal[0], dmp.goal[1], 'gx', mew=3)
    for obstacle in obstacles:
        plot_obs, = plt.plot(obstacle[0], obstacle[1], 'rx', mew=3)
    plot_path, = plt.plot(y_track[:,0], y_track[:, 1], 'b', lw=2)
    plt.title('DMP system - obstacle avoidance')

plt.axis('equal')
plt.xlim([-1.1,1.1])
plt.ylim([-1.1,1.1])
plt.show()
