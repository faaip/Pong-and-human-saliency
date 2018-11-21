from __future__ import print_function

import sys
import numpy as np
import gym
import time
import json
import datetime
import pickle

env = gym.make('Pong-v4')  # if len(sys.argv)<2 else sys.argv[1]

timestamp = datetime.datetime.now().strftime('%m_%d_%H%M')

if not hasattr(env.action_space, 'n'):
    raise Exception('Keyboard agent only supports discrete action spaces')
ACTIONS = env.action_space.n
SKIP_CONTROL = 0    # Use previous control decision SKIP_CONTROL times, that's how you
# can test what skip is still usable.
TIME_LIMIT = 60  # in seconds
human_agent_action = 0
human_wants_restart = False
human_sets_pause = False


def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause
    if key == 0xff0d:
        human_wants_restart = True
    if key == 32:
        human_sets_pause = not human_sets_pause
    a = int(key - ord('0'))
    if key == 65362:
        human_agent_action = 2
    elif key == 65364:
        human_agent_action = 3
    elif a <= 0 or a >= ACTIONS:
        return
    else:
        human_agent_action = a


def key_release(key, mod):
    global human_agent_action
    a = int(key - ord('0'))
    if (human_agent_action == a) or (key in set([65364, 65362])):
        human_agent_action = 0
    if a <= 0 or a >= ACTIONS:
        return


env.render()
env.unwrapped.viewer.window.on_key_press = key_press
env.unwrapped.viewer.window.on_key_release = key_release


def rollout(env):
    global human_agent_action, human_wants_restart, human_sets_pause
    human_wants_restart = False
    obser = env.reset()
    skip = 0
    total_reward = 0
    total_timesteps = 0
    start = time.time()
    indexes = 0
    frames = np.zeros((1000, 210, 160, 3), dtype=np.uint8)
    time_stamps = np.zeros([1000, 1], dtype=np.float64)
    while (time.time()-start) < TIME_LIMIT:
        if not skip:
            a = human_agent_action
            total_timesteps += 1
            skip = SKIP_CONTROL
        else:
            skip -= 1

        obser, r, done, info = env.step(a)

        # insert timestamp in array
        frames[indexes] = obser
        time_stamps[indexes] = time.time()
        print(time_stamps)
        indexes = indexes + 1

        if r != 0:
            print("reward %0.3f" % r)
        total_reward += r
        window_still_open = env.render()
        if window_still_open == False:
            return False
        if done:
            break
        if human_wants_restart:
            break
        while human_sets_pause:
            env.render()
            time.sleep(0.033)
        time.sleep(0.033)

    env.close()

    # slice and save np array
    frames = frames[:indexes]
    time_stamps = time_stamps[:indexes]
    with open('file_' + timestamp+'.pickle', 'wb') as f:
        pickle.dump([frames, time_stamps], f)
    # np.save('file_' + timestamp, frames)

    print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))


print("ACTIONS={}".format(ACTIONS))
print("Press keys 1 2 3 ... to take actions 1 2 3 ...")
print("No keys pressed is taking action 0")


window_still_open = rollout(env)
