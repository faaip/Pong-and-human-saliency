# Visualizing and Understanding Atari Agents | Sam Greydanus | 2017 | MIT License

from __future__ import print_function
#import warnings ; warnings.filterwarnings('ignore') # mute warnings, live dangerously

import matplotlib.pyplot as plt
import matplotlib as mpl ; mpl.use("Agg")
import matplotlib.animation as manimation
from tqdm import tqdm
import gym, os, sys, time, argparse
import pickle
import datetime
sys.path.append('..')
from exp_func import *

def make_agent_movie(env_name, checkpoint='*.tar', num_frames=20, first_frame=0, resolution=75, \
                save_dir='../data/saliency_agent/movies/', density=5, radius=5, prefix='default', overfit_mode=False, human_data_path = '', load_dir = ''):
    
    # set up dir variables and environment
    load_dir = '../data/model_agent/'
    meta = get_env_meta(env_name)
    env = gym.make(env_name)# if not overfit_mode else OverfitAtari(env_name, load_dir+'expert/', seed=0) # make a seeded env

    # set up agent
    model = NNPolicy(num_actions=env.action_space.n, channels = 1, memsize = 256)
    model.try_load(load_dir, checkpoint)

    # get a rollout of the policy
    movie_title = "{}-{}.mp4".format(prefix, datetime.datetime.now().strftime('%m-%d-%H-%M'))
    torch.manual_seed(0)
    if len(human_data_path) > 0:
        with open(human_data_path, 'rb') as f:
            frames  = pickle.load(f)[0]
        history = rollout_human(model,frames, num_frames)
    else:
        print('\tmaking movie "{}" using checkpoint at {}{}'.format(movie_title, load_dir, checkpoint))
        max_ep_len = first_frame + num_frames + 1
        history = rollout(model, env, max_ep_len=max_ep_len)


    # make the movie!
    start = time.time()
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title=movie_title, pad_inches = 0,artist='greydanus', comment='Saliency in the Arcade')
    writer = FFMpegWriter(fps=18, metadata=metadata)
    
    prog = '' ; total_frames = len(history['ins'])
    f = plt.figure(figsize=[6, 6*1.3], dpi=resolution)
    plt.axis('off')

    ax = f.add_subplot(1,1,1)
    f.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    l1 = ax.imshow(np.zeros((210,160,3)))
    ax.set_title(env_name, fontsize =15)

    with writer.saving(f, os.path.join(save_dir,movie_title), resolution):
        for i in tqdm(range(len(history['ins']))):
            ix = first_frame+i
            if ix < total_frames: # prevent loop from trying to process a frame ix greater than rollout length
                frame = history['ins'][ix].squeeze().copy()
                #actor_saliency = score_frame(model, history, ix, radius, density, interp_func=occlude, mode='actor')
                #critic_saliency = score_frame(model, history, ix, radius, density, interp_func=occlude, mode='critic')
            
                #frame = saliency_on_atari_frame(actor_saliency, frame, fudge_factor=meta['actor_ff'], channel=2)
                #frame = saliency_on_atari_frame(critic_saliency, frame, fudge_factor=meta['critic_ff'], channel=0)

                l1.set_data(frame)
                writer.grab_frame()
                
    print('\nfinished.')

# user might also want to access make_movie function from some other script
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-e', '--env', default='Pong-v0', type=str, help='gym environment')
    parser.add_argument('-d', '--density', default=5, type=int, help='density of grid of gaussian blurs')
    parser.add_argument('-r', '--radius', default=5, type=int, help='radius of gaussian blur')
    parser.add_argument('-f', '--num_frames', default=1000, type=int, help='number of frames in movie')
    parser.add_argument('-i', '--first_frame', default=0, type=int, help='index of first frame')
    parser.add_argument('-dpi', '--resolution', default=75, type=int, help='resolution (dpi)')
    parser.add_argument('-s', '--save_dir', default='../data/saliency_agent/movies/', type=str, help='dir to save agent logs and checkpoints')
    parser.add_argument('-l', '--load_dir', default='../data/model_agent/', type=str, help='dir checkpoints')
    parser.add_argument('-p', '--prefix', default='default',required = True, type=str, help='prefix to help make video name unique, number participant')
    parser.add_argument('-c', '--checkpoint', default='*.tar', type=str, help='checkpoint name (in case there is more than one')
    parser.add_argument('-o', '--overfit_mode', default=False, type=bool, help='analyze an overfit environment (see paper)')
    parser.add_argument('-x', '--human_data', default='', type=str, help='analyze an overfit environment (see paper)')
    args = parser.parse_args()

    make_agent_movie(args.env, args.checkpoint, args.num_frames, args.first_frame, args.resolution,
        args.save_dir, args.density, args.radius, args.prefix, args.overfit_mode, args.human_data, args.load_dir)
