import pickle, glob, re, os, datetime, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from exp_func import *

parser = argparse.ArgumentParser(description = None)
parser.add_argument('-a','--agent', default = '../data/model_agent/', type = str, help = 'Where are the weights of the policy')
parser.add_argument('-p','--part_dir', default = '../../Participant_data/', type = str, help = 'Folder contain the particants')

args = parser.parse_args()

# Get the Policy and download the agents model

model = NNPolicy(num_actions=6, channels = 1, memsize = 256)
model.try_load(args.agent)
meta = {}
meta['critic_ff'] = 600 ; meta['actor_ff'] = 500
radius = 5
density = 5


all_participants = sorted(glob.glob(os.path.join(args.part_dir, '*')))

for part in all_participants:
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
  tobi = tobi.dropna(subset = [gaze_columns])

  tobi['GazePointX (ADCSpx)'] = np.clip(tobi['GazePointX (ADCSpx)'],0,1279)
  tobi['GazePointY (ADCSpx)'] = np.clip(tobi['GazePointY (ADCSpx)'],0,1023)

  # Translate the timestamps to POSIX time
  recorddates = tobi['RecordingDate'] +' '+ tobi['LocalTimeStamp']

  tobi_timestamps = recorddates.apply(lambda x: datetime.datetime.strptime(x,'%d-%m-%Y %H:%M:%S.%f').timestamp()).values


  # Run over the pickle files and make the video
  for num, p_file in enumerate(pickle_files):
    print('Parsing: {}'.format(p_file), flush = True)
    with open(p_file, 'rb') as f:
      files = pickle.load(f)
    
    python_timestamps = np.sort(files[1].reshape(-1))
    frames = files[0]

    
    valid_times_bool = (tobi_timestamps > python_timestamps[0]) & (tobi_timestamps < python_timestamps[-1])
    valid_timestamps = tobi_timestamps[ valid_times_bool ]
    valid_tobi = tobi.iloc[valid_times_bool]
    
    connect = np.zeros(len(valid_timestamps), dtype = int)
    
    max_delta = 0
    
    for i, t in enumerate(valid_timestamps):
      differences = np.abs(t-python_timestamps)
      ind = np.argmin(differences)
      max_delta = max(max_delta,differences[ind])
      connect[i] = ind.astype('uint')

    
    gaze_points = valid_tobi[gaze_columns]
    # Make the human saliency map.
    human_saliency = np.zeros(frames.shape[:3], dtype = 'uint8')
    g_spots = gaze_points.values
    fix_points = (np.array([160,210])*g_spots/np.array([1280,1024])).T.astype(int)
    human_saliency[connect,fix_points[1],fix_points[0]] = 1
    

    agent_saliency = np.zeros(frames.shape, dtype = 'uint8')
    history = rollout_human(model,frames, len(frames))

    for ix in tqdm(range(len(history['ins']))):
      actor = score_frame(model, history, ix, radius, density, interp_func=occlude, mode='actor')
      agent_saliency[ix] = saliency_on_atari_frame(actor, agent_saliency[ix], fudge_factor=meta['actor_ff'], channel=0)
    
    agent_saliency = agent_saliency[:,:,:,0]
    
    name = 'Play_round_'+str(num)+'.pickle'
    with open(os.path.join(part_1,name), 'wb') as f:
      pickle.dump([frames,human_saliency, agent_saliency], f)

    print('Saved pickle file: {}'.format(name))
