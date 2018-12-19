# %%
import pickle
import os
import numpy as np
import time
import pandas as pd
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# %%
rootdir = '/Users/au550101/Dropbox (Personal)/Cognitive_science_3/Participant_data'


def get_static_aoi(human_fixation_frame):
    """ 
    input: frame containing human fixation
    returns 0: score board, 1: opponents, 2: center area, 3: player's area None: no AOI
    """
    point = np.mean(np.where(human_fixation_frame == 1), axis=1)
    if np.isnan(point[0]):
        return 4  # no fixation point
    if point[0] < 33:
        return 0  # score board
    if point[1] < 40:
        return 1  # opponents area
    if point[1] < 120:
        return 2  # central area
    return 3  # player's area


def make_the_pie_plot(aois, static):
    """
    This function creates a pie chart.
    :param aois: dynamic or static aois
    :param static: bool whether pie is static or not
    :return: None
    """
    unique, counts = np.unique(aois, return_counts=True)
    percentages = np.asarray(counts/np.sum(counts)*100).T

    fig1, ax1 = plt.subplots()
    if static:
        labels = ['score board', 'opponent', 'central', 'player']
        ax1.set_title('Static AOIs')
    else:
        labels = ['ball', 'opponent', 'player']
        ax1.set_title('Dynamic AOIs')
    ax1.pie(percentages[:-1], labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')
    plt.figtext(0.99, 0.01, 'No fixation percentage: {0:.2f} %'.format(
        percentages[-1]), horizontalalignment='right')
    plt.show()


def make_beauty_plot(fixation_human, frames):
    """
    This function creates a plot of static AOI
    :param fixation_human: human fixation array
    :param frames: all the frames
    :return: None
    """
    static_aois = np.array([get_static_aoi(fixation_human[i])
                            for i in range(fixation_human.shape[0])])
    unique, counts = np.unique(static_aois, return_counts=True)
    # percentages (disregarding no fixation)
    percentages = np.asarray(counts/np.sum(counts)*100).T

    fig1, ax1 = plt.subplots()
    ax1.imshow(frames[0])

    rects = [[(0, 0), 160, 33, 'purple'],
             [(0, 33), 40, 197, 'blue'],
             [(40, 33), 80, 197, 'pink'],
             [(120, 33), 40, 197, 'green']
             ]

    for idx, rect in enumerate(rects):
        r = patches.Rectangle(rect[0], rect[1], rect[2], linewidth=1,
                              edgecolor=None, facecolor=rect[3], alpha=.5)
        ax1.set_title('Static AOIs')
        ax1.add_patch(r)
        ax1.text(rect[0][0] + rect[1]/2, rect[0][1] + rect[2]/2,
                 '{0:.2f} %'.format(percentages[idx]), ha='center')
        ax1.set_yticklabels([])
        ax1.set_xticklabels([])

    plt.figtext(0.99, 0.01, 'No fixation percentage: {0:.2f} %'.format(
        percentages[4]), horizontalalignment='right')
    plt.show()


def get_dynamic_aoi(human_fixation_frame, frame):
    fix_point = np.mean(np.where(human_fixation_frame == 1), axis=1)

    aois = [[0, (236, 236, 236)],  # ball
            [1, (213, 130, 74)],  # opponent
            [2, (92, 186, 92)]]  # paddle

    for aoi in aois:
        p = np.where(frame[34:194, :] == aoi[1])
        p = np.mean(p, axis=1)[:2]
        dist = np.linalg.norm(p-fix_point)

        if dist < 30:  # TODO: set meaningful threshold
            return aoi[0]

    return 3  # None


# %%
def get_aoi_row(file_path):
    row = {}
    try:
        row['participant_no'] = int(file_path.split('_')[-1])
    except ValueError:
        return None

    # values for calculating totals
    vals = {}

    # Iterate through rounds
    for subdir, dirs, files in os.walk(file_path):
        for file in files:
            if ('Play' in file or 'Watch' in file) is False:
                continue

            # Info on round
            round_type = file.split('_')[0]

            # Load them pickles
            file = open(file_path + '/' + file, 'rb')
            object_file = pickle.load(file)
            file.close()
            frames, fixation_human, agent_saliency = object_file

            # Calculate static aois
            static_aois = np.array([get_static_aoi(fixation_human[i])
                                    for i in range(fixation_human.shape[0])])
            static_unique, static_counts = np.unique(
                static_aois, return_counts=True)

            # Calculate dynamic aois
            dynamic_aois = np.array([get_dynamic_aoi(fixation_human[i], frames[i])
                                     for i in range(fixation_human.shape[0])])
            dynamic_unique, dynamic_counts = np.unique(
                dynamic_aois, return_counts=True)
            dynamic_percentages = np.asarray(
                dynamic_counts/np.sum(dynamic_counts)*100).T

            # Insert totalt values for later calculation
            try:
                vals[round_type + '_static_counts'] + static_counts
                vals[round_type + '_dynamic_counts'] + dynamic_counts
            except KeyError:
                vals[round_type + '_static_counts'] = static_counts
                vals[round_type + '_dynamic_counts'] = dynamic_counts
            except ValueError:
                pass

    watch_static_percentages = np.asarray(
        vals['Watch_static_counts']/np.sum(vals['Watch_static_counts'])*100).T
    play_static_percentages = np.asarray(
        vals['Play_static_counts']/np.sum(vals['Play_static_counts'])*100).T

    watch_dynamic_percentages = np.asarray(
        vals['Watch_dynamic_counts']/np.sum(vals['Watch_dynamic_counts'])*100).T
    play_dynamic_percentages = np.asarray(
        vals['Play_dynamic_counts']/np.sum(vals['Play_dynamic_counts'])*100).T

    # static separate percentages
    for idx, label in enumerate(['score board', 'opponent', 'central', 'player']):
        row['Watch_' + label + '_static'] = watch_static_percentages[idx]
        row['Play_' + label + '_static'] = play_static_percentages[idx]

    for idx, label in enumerate(['ball', 'opponent', 'player']):
        row['Watch_' + label + '_dynamic'] = watch_dynamic_percentages[idx]
        row['Play_' + label + '_dynamic'] = play_dynamic_percentages[idx]

    row['Watch_total_time_fixation_static'] = 100-watch_static_percentages[-1]
    row['Play_total_time_fixation_static'] = 100-play_static_percentages[-1]
    row['Watch_total_time_fixation_dynamic'] = 100 - \
        watch_dynamic_percentages[-1]
    row['Play_total_time_fixation_dynamic'] = 100-play_dynamic_percentages[-1]

    return row


rows = []
# start iterating
for subdir, dirs, files in os.walk(rootdir):
    row = get_aoi_row(subdir)
    if row is not None:
        rows.append(row)


# %%
df = pd.DataFrame.from_dict(rows)
df = df.set_index('participant_no')
df

# %%
df.to_csv(data.csv, sep='\t')


# %%
