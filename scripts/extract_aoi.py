# %%
import pickle
import numpy as np
import time
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# %%
file = open("data/Participant_1/Play_round_0.pickle", 'rb')
object_file = pickle.load(file)
file.close()
frames, fixation_human, agent_saliency = object_file


# %%
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
        dist = numpy.linalg.norm(p-fix_point)

        if dist < 25:  # TODO: set meaningful threshold
            return aoi[0]

    return 3 # None


# static
# %%
static_aois = np.array([get_static_aoi(fixation_human[i])
                        for i in range(fixation_human.shape[0])])
make_the_pie_plot(static_aois, True)
make_beauty_plot(fixation_human, frames)

# dynamic
aois = np.array([get_dynamic_aoi(fixation_human[i], frames[i])
                 for i in range(fixation_human.shape[0])])
make_the_pie_plot(aois, False);


# %%
