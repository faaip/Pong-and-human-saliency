import pickle, re, torch, glob, os
import datetime
import numpy as np
import pandas as pd
from scipy.misc import imresize
import skimage
import matplotlib.pyplot as plt


def make_datasets(frames, selected, connect, name):
    prepro = lambda img: imresize(img[35:195].mean(2), (80, 80)).astype(np.float32).reshape(80, 80) / 255.
    prepro_label = lambda img: skimage.filters.gaussian(imresize(img[35:195], (80, 80)).astype(np.float32), sigma=8)

    print(frames.shape)

    data = np.zeros((len(frames), 80, 80))
    labels = np.zeros((len(frames), 80, 80))
    wanted_id = 0

    for idx in range(len(frames)):
        frame = frames[idx]
        # downsample
        frame = prepro(frame)

        g_spots = selected.iloc[connect == idx].values

        if len(g_spots) > 0:
            fix_points = (np.array([160, 210]) * g_spots / np.array([1280, 1024])).T.astype(int)

            label_frame = np.zeros((210, 160))

            label_frame[fix_points[1], fix_points[0]] = 1
            label_frame = prepro_label(label_frame)
            data[wanted_id] = frame
            labels[wanted_id] = label_frame
            wanted_id += 1

    data = data[:wanted_id]
    labels = labels[:wanted_id]
    print(data.shape, labels.shape)

    with open(name + '.pickle', 'wb') as f:
        pickle.dump([data, labels], f)


# Get the files in the directory
part_dir = 'Participant_5'
participant = glob.glob(os.path.join(part_dir, '*'))

# Get the files in the directory


# Get the tobi file / pickle files
pickle_files = []
for i in participant:
    if len(re.findall('.*.tsv', i)) > 0:
        tobi_file = i
    if len(re.findall('.*.pickle', i)) > 0:
        pickle_files.append(i)

pickle_files.sort()

gaze_columns = ['GazePointX (ADCSpx)', 'GazePointY (ADCSpx)']

# read tobi file, delete rows with nan in the gaze columns
tobi = pd.read_csv(tobi_file, delimiter='\t')
tobi = tobi.dropna(subset=[gaze_columns])

tobi['GazePointX (ADCSpx)'] = np.clip(tobi['GazePointX (ADCSpx)'], 0, 1279)
tobi['GazePointY (ADCSpx)'] = np.clip(tobi['GazePointY (ADCSpx)'], 0, 1023)
# Translate the timestamps to POSIX time
recorddates = tobi['RecordingDate'] + ' ' + tobi['LocalTimeStamp']

tobi_timestamps = recorddates.apply(lambda x: datetime.datetime.strptime(x, '%d-%m-%Y %H:%M:%S.%f').timestamp()).values

# Run over the pickle files and make the video
for num, p_file in enumerate(pickle_files):
    with open(p_file, 'rb') as f:
        files = pickle.load(f)

    python_timestamps = np.sort(files[1].reshape(-1))
    frames = files[0]

    valid_times_bool = (tobi_timestamps > python_timestamps[0]) & (tobi_timestamps < python_timestamps[-1])
    valid_timestamps = tobi_timestamps[valid_times_bool]
    valid_tobi = tobi.iloc[valid_times_bool]

    # connect = np.zeros(len(python_timestamps))
    connect = np.zeros(len(valid_timestamps))

    max_delta = 0
    # for i, t in enumerate(python_timestamps):
    for i, t in enumerate(valid_timestamps):
        # differences = np.abs(t-tobi_timestamps)
        differences = np.abs(t - python_timestamps)
        ind = np.argmin(differences)
        max_delta = max(max_delta, differences[ind])
        connect[i] = ind

    print(len(valid_timestamps), valid_tobi.shape, max_delta, flush=True)
    # selected = tobi.iloc(connect)[gaze_columns]

    selected = valid_tobi[gaze_columns]
    fps = int(np.round(1 / (python_timestamps[1:] - python_timestamps[:-1]).mean()))

    name = os.path.join('Data', part_dir + '_' + str(num))

    make_datasets(frames, selected, connect, name)
