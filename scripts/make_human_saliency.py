import pickle, argparse, time, datetime, os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation


'''
This script is meant to make a video of the human saliency
'''

now = datetime.datetime.now()

parser = argparse.ArgumentParser(description = None)
parser.add_argument('-d','--dir_path',type=str,default = '../data/tobi/', help = 'Path to data dir')
parser.add_argument('-r','--pickle',type=str, help= 'The name of the pickle file')
parser.add_argument('-t','--tobi',type=str, help= 'The name of the tobi file')
parser.add_argument('-p','--prefix',type=str,default = 'part',help='Prefix, participants')
parser.add_argument('-o','--output',type=str, default = '../data/saliency_human/movies/' ,help= 'Path of output file')

args = parser.parse_args()

pickle_file = os.path.join(args.dir_path,args.pickle)
tobi_file = os.path.join(args.dir_path,args.tobi)
resolution_tobi_screen = np.array([1280,1024])
resolution = 75


with open(pickle_file, 'rb') as f:
  files = pickle.load(f)

python_timestamps = files[1].reshape(-1)
frames = files[0]

tobi = pd.read_csv(tobi_file, delimiter = '\t')

recorddates = tobi['RecordingDate'] +' '+ tobi['LocalTimeStamp']
tobi_timestamps = recorddates.apply(lambda x: datetime.datetime.strptime(x,'%d-%m-%Y %H:%M:%S.%f').timestamp()).values

connect = np.zeros(len(python_timestamps))

for i, t in enumerate(python_timestamps):
  ind = np.argmin(np.abs(t-tobi_timestamps))
  connect[i] = ind

selected = tobi.iloc[connect][['FixationPointX (MCSpx)','FixationPointY (MCSpx)']]

frame_p_sec = int(np.round(1/ (python_timestamps[1:]-python_timestamps[:-1]).mean()))

# make the movie!
start = time.time()
FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Saliency movie', artist='Saliency', comment='atari-saliency-video')
writer = FFMpegWriter(fps=frame_p_sec, metadata=metadata)

prog = ''
total_frames = len(frames)
f = plt.figure(figsize=[6, 6*1.3], dpi=resolution)
ax = f.add_subplot(1,1,1)
ax.set_title('Name', fontsize =15)

output_file = os.path.join(args.output,args.prefix+now.strftime('-%m-%d-%H-%M.mp4'))

with writer.saving(f, output_file, resolution):
  for ix in tqdm(range(total_frames)):
    if ix < total_frames: # prevent loop from trying to process a frame ix greater than rollout length
      frame = frames[ix].squeeze().copy()
      fix_point_x , fix_point_y = np.array([160,210])*selected.iloc[ix].values/resolution_tobi_screen



      ax.imshow(frame)
      ax.scatter(fix_point_x,fix_point_y,c = 'red', s = 500, alpha = .5)
      writer.grab_frame()
      ax.clear()
