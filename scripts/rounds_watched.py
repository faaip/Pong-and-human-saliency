import pickle, glob, re, os, datetime, sys, gym, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from exp_func import *

parser = argparse.ArgumentParser(description = None)
parser.add_argument('-a','--agent', default = '../data/model_agent/', type = str, help = 'Where are the     weights of the policy')
parser.add_argument('-p','--part_dir', default = '../../Participant_data/', type = str, help = 'Folder     contain the particants')
parser.add_argument('-m','--movie_dir', default = '../experiment_movies/pickles/', type = str, help = 'Folder     contain the particants')

args = parser.parse_args()

# For the cleaning of the images we need the right colours, so we need to go to the gym
env=gym.make('Pong-v0')
pif = env.reset()
for _ in range(30):
  pif = env.step(0)

available_tuples = np.array(list(set(tuple(i) for i in pif[0].reshape(-1,3))), dtype =float)

def clean_frames(old_image, available_colours = available_tuples):
  background = [144,72,17]
  correct_shape = old_image.shape
  old = old_image.reshape(-1,3).astype(float)[:,np.newaxis,:]
  available_col = available_tuples[np.newaxis]
  
  dist = np.multiply(old-available_col,old-available_col).sum(axis = -1)

  best_fit = np.argmin(dist, axis = -1)
  
  new_image = available_colours[best_fit]
  
  new_image = new_image.reshape(correct_shape)
  new_image[:,34:194,:5] = background
  new_image[:,34:194,-5:] = background

  
  return new_image.astype('uint8')

# Get the Policy

model = NNPolicy(num_actions=6, channels = 1, memsize = 256)
model.try_load(args.agent)
meta = {}
meta['critic_ff'] = 600 ; meta['actor_ff'] = 500
radius = 5
density = 5


all_participants = sorted(glob.glob(os.path.join(args.part_dir,'*')))

for part in all_participants:
  print('-'*30)
  print(part)
  part_files = glob.glob(os.path.join(part,'*'))

  # Get the tobi file / pickle files
  pickle_files = []
  for i in part_files:
    if len(re.findall('.*.tsv',i)) > 0:
      tobi_file = i
    if len(re.findall('file.*.pickle',i)) > 0:
      pickle_files.append(i)

  pickle_files.sort()

  gaze_columns = ['GazePointX (ADCSpx)','GazePointY (ADCSpx)']
      
    
  # read tobi file, delete rows with nan in the gaze columns
  tobi = pd.read_csv(tobi_file, delimiter = '\t')
  tobi['GazePointX (ADCSpx)'] = np.clip(tobi['GazePointX (ADCSpx)'],0,1279)
  tobi['GazePointY (ADCSpx)'] = np.clip(tobi['GazePointY (ADCSpx)'],0,1023)

  movies = glob.glob(os.path.join(args.movie_dir , '*'))

  for num, movie in enumerate(sorted(movies)):
    print('Starting with: {}'.format(movie))
    with open(movie, 'rb') as f:
      frames = pickle.load(f)
    
    frames = clean_frames(frames)
    name = re.findall('(just_agent.*.)mp4',movie)[0]+'avi'

    tobi_gaze = tobi[tobi['MediaName'] == name][gaze_columns]
    tobi_gaze = (np.array([160,210])*tobi_gaze/np.array([1280,1024]))
    gaze_partitioned = np.array_split(tobi_gaze,len(frames))

    human_saliency = np.zeros(frames.shape[:3], dtype = 'uint8')
    for j, gaze in enumerate(gaze_partitioned):
      gazes = gaze.dropna().T.values.astype('int')
      human_saliency[[j],gazes[1],gazes[0]] = 1
    
    agent_saliency = np.zeros(frames.shape, dtype = 'uint8')

    history = rollout_human(model,frames, len(frames))
    print('Cleaned Images, starting with agent saliency ...', flush = True)
    for ix in tqdm(range(len(history['ins']))):
      actor = score_frame(model, history, ix, radius, density, interp_func=occlude, mode='actor')
      agent_saliency[ix] = saliency_on_atari_frame(actor, agent_saliency[ix], fudge_factor=meta['actor_ff'], channel=0)
    
    agent_saliency = agent_saliency[:,:,:,0]

    
    out_name = 'Watch_round_'+str(num)+'.pickle'
    with open(os.path.join(part_1,out_name), 'wb') as f:
      pickle.dump([frames,human_saliency, agent_saliency], f)

    print('Saved file: {}'.format(out_name))

