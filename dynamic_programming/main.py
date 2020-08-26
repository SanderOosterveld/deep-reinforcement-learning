import math

import numpy as np

from environments import ContinuousUpswingPendulum as PendulumEnvironment
import time
import multiprocessing as mp
from itertools import count
import matplotlib.pyplot as plt
np.random.seed(1)


def build_grid(steps, range_velocity=(-3 * math.pi, 3 * math.pi)):
    """
    Build a of values such that we can do: values[n][m] where the first index is the nth angle (n= steps = 2pi) and
    second index is the
    :param steps:
    :param range_velocity:
    """
    range_angle = (0, 2 * math.pi)
    angles = np.linspace(range_angle[0], range_angle[1], num=steps)

    velocities = np.linspace(range_velocity[0], range_velocity[1], num=steps)
    angles = np.matmul(angles.reshape((steps, 1)), np.ones((1, steps)))
    velocities = np.matmul(np.ones((steps, 1)), velocities.reshape((1, steps)))

    angles = angles.reshape((steps, steps, 1))

    velocities = velocities.reshape((steps, steps, 1))
    values = np.concatenate((angles, velocities), axis=2)

    return values


def add_actions(grid: np.ndarray, action_space=(-1, 0, 1)):
    shape = grid.shape
    new_shape = (shape[0], shape[1], len(action_space), shape[2])
    new_grid = np.zeros(new_shape)
    for i in range(len(action_space)):
        new_grid[:, :, i, :] = grid
    return new_grid


environment = PendulumEnvironment()
N = 700

action_space = (-1, 0, 1)
num_actions = len(action_space)
grid = build_grid(N)
new_grid = add_actions(grid, action_space=action_space)
size = new_grid.shape
final_grid = new_grid.reshape((*size[0:3], 1, size[3]))
index_grid = np.concatenate((final_grid, np.zeros(final_grid.shape)), axis=3)
final_grid = np.concatenate((final_grid, np.zeros(final_grid.shape)), axis=3)
final_grid_size = final_grid.shape


def build_next_state(x):
    environment = PendulumEnvironment()
    output = np.zeros((1, size[1], size[2], 2))
    for j in range(size[1]):
        for k in range(size[2]):
            environment.state = final_grid[x, j, k, 0]
            environment.step(action_space[k])
            output[0, j, k, 0] = environment.simulator.get_real_angle()
            output[0, j, k, 1] = environment.state[1]
    return output, x


def get_new_states(chunksize=16):
    pool = mp.Pool()

    for res, arg in pool.imap(build_next_state, range(N), chunksize=chunksize):
        final_grid[arg, :, :, 1, :] = res
        if arg % 100 == 0:
            print(arg)
    pass


range_angle = (0, 2 * math.pi)
angles = np.linspace(range_angle[0], range_angle[1], num=N)
range_velocity = (-3 * math.pi, 3 * math.pi)
velocities = np.linspace(range_velocity[0], range_velocity[1], num=N)


def find_closest_index(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx == len(array):
        print("state_outside_range: %f"%value)
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        return idx - 1
    else:
        return idx


def build_next_index(x):
    output = np.zeros((1, size[1], size[2], 2))
    for j in range(size[1]):
        for k in range(size[2]):
            value = final_grid[x, j, k, 1, 0]
            angle_index = find_closest_index(angles, final_grid[x, j, k, 1, 0])
            velocity_index = find_closest_index(velocities, final_grid[x, j, k, 1, 1])
            output[0, j, k, 0] = angle_index
            output[0, j, k, 1] = velocity_index
    return output, x


def get_next_state_index(chunksize=16):
    pool = mp.Pool()

    for res, arg in pool.imap(build_next_index, range(N), chunksize=chunksize):
        index_grid[arg, :, :, 1, :] = res
        if arg % 100 == 0:
            print(arg)


reward_matrix = np.zeros((N, N, num_actions))


def build_reward(x):
    output = np.zeros((1, N, num_actions))
    environment = PendulumEnvironment()
    for j in range(N):
        for k in range(num_actions):
            next_state_index = index_grid[x, j, k, 1].astype('int')
            next_state = index_grid[next_state_index[0], next_state_index[1]][k][0]
            # print("j: %i, k: %i, next_state: %s, index %s"%(j, k, next_state, next_state_index))
            environment.state = next_state
            output[0, j, k] = environment.get_reward()
    return output, x


def get_reward_matrix(chunksize=16):
    pool = mp.Pool()

    for res, arg in pool.imap(build_reward, range(N), chunksize=chunksize):
        reward_matrix[arg, :, :] = res
        if arg % 100 == 0:
            print(arg)


Q_matrix = np.zeros((N, N, num_actions))
Q_new = np.zeros((N, N, num_actions))
A_matrix = np.zeros((N, N))

gamma = 0.99


def get_Q_matrix() -> np.ndarray:
    Q_matrix = np.zeros((N, N, num_actions))
    for i in range(1000):
        original_shape = Q_matrix.shape
        linear_Q_matrix = Q_matrix.reshape((-1))
        indexs = index_grid[:, :, :, 1].reshape(num_actions * N * N, -1).transpose().astype('int')

        Q_new = (reward_matrix.reshape(-1) + gamma * np.amax(Q_matrix[indexs[0], indexs[1]], 1))
        diff = np.amax(np.absolute(linear_Q_matrix - Q_new))
        if i % 10 == 0:
            print(diff)
        if diff < 10 ** -20:
            print(diff)
            print(i)
            Q_matrix = Q_new.reshape(original_shape)

            break
        Q_matrix = Q_new.reshape(original_shape)
    return Q_matrix


def run(policy_matrix, runs = 320):
    environment = PendulumEnvironment()
    environment.evaluated_init()
    # environment.state = [3,0]
    total_reward = environment.get_reward()
    all_angles = np.zeros(runs)
    all_velocties = np.zeros(runs)
    for i in count():
        state = environment.state
        all_angles[i] = environment.simulator.get_real_angle()
        all_velocties[i] = state[1]
        angle_index = find_closest_index(angles, environment.simulator.get_real_angle())
        vel_index = find_closest_index(velocities, state[1])
        action = action_space[policy_matrix[angle_index][vel_index]]
        environment.step(action)
        total_reward += environment.get_reward()
        if environment.done:
            break
    return total_reward, all_angles, all_velocties

# def noisy_run(policy_matrix, state_noise=4, runs=200):
#     environment = NoisyPendulumEnvironment(noise_angle=state_noise)
#     average_angles_all = np.zeros((runs,10))
#     average_velocities_all =np.zeros((runs,10))
#     average_rewards_all = np.zeros(10)
#
#     for j in range(10):
#         environment.np_init(np.array([0,0]))
#         total_reward = environment.get_reward()
#         total_reward = 0
#         all_angles = np.zeros(runs)
#         all_velocties = np.zeros(runs)
#
#         for i in range(runs):
#             state = environment.state
#             state[0] = environment.get_real_angle()
#             all_angles[i] = state[0]
#             all_velocties[i] = state[1]
#             angle_index = find_closest_index(angles, state[0])
#             vel_index = find_closest_index(velocities, state[1])
#             action = action_space[policy_matrix[angle_index][vel_index]]
#             environment.step(action)
#             total_reward += environment.get_reward()
#
#         average_rewards_all[j] = total_reward
#         average_angles_all[:,j] = all_angles
#         average_velocities_all[:,j] = all_velocties
#     print(average_rewards_all)
#     print(np.mean(average_rewards_all))
#     print(np.mean(average_velocities_all, 0))
#     print(np.mean(average_velocities_all, 1))
#
#     return total_reward, all_angles, all_velocties

# def noisy_show(policy_matrix)
# def show(policy_matrix, runs = 200):
#     state_plotter = StatePlotter()
#     environment = PendulumEnvironment()
#     environment.np_init(np.array([0,0]))
#     total_reward = environment.get_reward()
#     for i in range(runs):
#         state = environment.state
#         state[0] = environment.get_real_angle()
#         angle_index = find_closest_index(angles, state[0])
#         vel_index = find_closest_index(velocities, state[1])
#         action = action_space[policy_matrix[angle_index][vel_index]]
#         environment.step(action)
#         total_reward+=environment.get_reward()
#         state_plotter.title = "Reward: %f, Action: %f" % (total_reward, action)
#         state_plotter.update(environment.state_object)
#
# import matplotlib.pyplot as plt
import os
if __name__ == '__main__':
    dyn_directory = os.path.dirname(os.path.realpath(__file__))
    # get_new_states()
    # final_grid.tofile(os.path.join(dyn_directory,'grid.dat'))
    # # final_grid = np.fromfile('small_dt-500runs.dat').reshape(final_grid.shape)
    # get_next_state_index()
    # index_grid.tofile(os.path.join(dyn_directory,'index_grid.dat'))
    # # index_grid = np.fromfile('index_grid.dat').reshape(index_grid.shape)
    # get_reward_matrix()
    # reward_matrix.tofile(os.path.join(dyn_directory,'reward_matrix.dat'))
    #

    # Q_matrix = get_Q_matrix()
    # Q_matrix.tofile(os.path.join(dyn_directory,'q-matrix.dat'))
    Q_matrix = np.fromfile(os.path.join(dyn_directory,'q-matrix.dat')).reshape(Q_matrix.shape)    #

    # final_grid = np.fromfile(os.path.join(dyn_directory,'grid.dat')).reshape(final_grid.shape)
    # index_grid = np.fromfile(os.path.join(dyn_directory,'index_grid.dat')).reshape(index_grid.shape)
    # reward_matrix = np.fromfile(os.path.join(dyn_directory,'reward_matrix.dat')).reshape(reward_matrix.shape)
    # Q_matrix = get_Q_matrix()
    # Q_matrix.tofile(os.path.join(dyn_directory, 'q-matrix.dat'))
    #get_reward_matrix()
    # reward_matrix.tofile('1000runs_reward.dat')
    #
   # Q_matrix = get_Q_matrix()
    # Q_matrix.tofile('1000runs_Q_function.dat')
    value_matrix = np.amax(Q_matrix, 2)
    policy_matrix = np.argmax(Q_matrix, 2)
    np.savetxt(os.path.join(dyn_directory, 'policy.csv'), policy_matrix, delimiter=',', fmt='%.10f')
    np.savetxt(os.path.join(dyn_directory, 'value.csv'), value_matrix, delimiter=',', fmt='%.10f')

    #policy_matrix = 2*np.ones(policy_matrix.shape, dtype=np.int)
    # print(final_grid)
    # print(index_grid)
    # print(reward_matrix)


    data = run(policy_matrix)
    print(data)
    np.savetxt(os.path.join(dyn_directory, 'angles.csv'), data[1], delimiter=',', fmt='%.10f')
    np.savetxt(os.path.join(dyn_directory, 'velocities.csv'), data[2], delimiter=',', fmt='%.10f')

    # print(run(policy_matrix))


    # print(environment.get_reward())
    policy_matrix = np.roll(policy_matrix, round(policy_matrix.shape[0]/2), axis=0)
    value_matrix = np.roll(value_matrix, round(policy_matrix.shape[0]/2), axis=0)

    # print(Q_matrix[0:10, 495:505])
    # print(final_grid[0:10,495:505,:,0,0])


    fix, ax = plt.subplots()
    im = ax.imshow(policy_matrix, aspect='auto')

    fix, ax = plt.subplots()
    im = ax.imshow(value_matrix, aspect='auto')
    plt.show()


    # environment.np_init((6.12, -3.532))
    # print(environment.get_real_angle())
    # print(environment.get_reward())
