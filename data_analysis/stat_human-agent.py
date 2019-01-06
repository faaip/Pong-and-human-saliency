# This file contains code for performing statistical analysis on both human eye-tracking data
# and the agent saliency data for later comparison. Data in pickle files has to be in same directory,
# folder structure as such: Participant{}/Play_round_{} for playing data, Participant{}/Watch_round_{}
# for watching data.
# Code for displaying example frames is commented out for continuity of code, uncomment for viewing.


print('Run statistical script. Make sure pickle data is in current directory as instructed.')


# import libs and modules
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from numpy.random import randint
import scipy as sp
from scipy.stats import entropy
from scipy.stats import ttest_ind_from_stats
from scipy.stats import ttest_ind
import pickle
import matplotlib.pyplot as plt
import cv2
from cv2 import resize

print('Loaded libraries and modules')


# settings for loading data
nr_participants = 13
nr_watchrounds = 4
nr_playrounds = 5


# extract human playing data and agent's corresponding data
playhuman = []
playagent = []

print('Start loading playing data from pickle files.')

for j in range(nr_participants):
    for i in range(nr_playrounds):
        with open('Participant{0}/Play_round_{1}.pickle'.format(j+1, i), 'rb') as f:
            x = pickle.load(f)
            playhuman.append(x[1])
            playagent.append(x[2])

print('Playing data successfully loaded.')


# extract human watching data and agent's corresponding data
watchhuman = []
watchagent = []

print('Start loading watching data from pickle files.')

for j in range(nr_participants):
    for i in range(nr_watchrounds):
        with open('Participant{0}/Watch_round_{1}.pickle'.format(j+1, i), 'rb') as f:
            x = pickle.load(f)
            watchhuman.append(x[1])
            watchagent.append(x[2])

print('Watching data successfully loaded.')


# Set all fixation points outside playing field to 0
def get_playingfield(data):
    for j in range(len(data)):
        for i in range(len(data[j])):
            indices = np.where(data[j][i]==1)
            scoreboard = np.where(indices[0]<35)
            bottom = np.where(indices[0]>195)

            for sc in scoreboard[0]:
                data[j][i][indices[0][sc], indices[1][sc]] = 0

            for sc in bottom[0]:
                data[j][i][indices[0][sc], indices[1][sc]] = 0
    return data


print('Set points outside playing field to 0.')
ph = get_playingfield(playhuman)
pa = get_playingfield(playagent)
wh = get_playingfield(watchhuman)
wa = get_playingfield(watchagent)



##
# Uncomment next part (lines 92 - 134) for viewing example frames and their KL-divergence
##


# # View example frames side by side
# ind1 = 150
# r1 = 0
# ind2 = 261
# r2 = 3
# ind3 = 500
# r3 = 5


# frame1 = ph[r1][ind1] * 100
# frame1 = cv2.GaussianBlur(frame1,(31,31),8)

# frame2 = ph[r2][ind2] * 100
# frame2 = cv2.GaussianBlur(frame2,(31,31),8)

# frame3 = ph[r3][ind3] * 100
# frame3 = cv2.GaussianBlur(frame3,(31,31),8)

# fig = plt.figure(figsize=(12,15) )
# ax1 = plt.subplot(221)
# plt.imshow(frame2, cmap='gray_r')
# ax1.set_title('Human saliency', fontsize=25)
# ax1.set_xticks([])
# ax1.set_yticks([])
# plt.ylabel('Highly similar', fontsize=25)
# ax2 = plt.subplot(222)
# plt.imshow(pa[r2][ind2], cmap='gray_r')
# ax2.set_title('Agent saliency', fontsize=25)
# ax2.set_xticks([])
# ax2.set_yticks([])
# ax3 = plt.subplot(223)
# plt.imshow(frame3, cmap='gray_r')
# plt.ylabel('Highly dissimilar', fontsize=25)
# ax3.set_xticks([])
# ax3.set_yticks([])
# ax4 = plt.subplot(224)
# plt.imshow(pa[r3][ind3], cmap='gray_r')
# ax4.set_xticks([])
# ax4.set_yticks([])

# # Get KL-divergence for each human-agent pairing
# print('KL-divergence for first pair: ', entropy(frame2.reshape(-1), pa[r2][ind2].reshape(-1)))
# print('KL-divergence for second pair: ', entropy(frame3.reshape(-1), pa[r3][ind3].reshape(-1)))




# Calculate KL-divergence for each human-agent frame pair, return array of all values (outputs mean and std)
# include rs=True for resizing to original size

def get_ent(human, agent, rs=False):
    
    all_entropies = []
    for j in range(len(human)):
        for i in range(len(human[j])):
            frame = human[j][i]
            indices = np.where(frame==1)

            if indices[0].any():
                frame = frame * 100
                frame = cv2.GaussianBlur(frame,(31,31),8)
                
                comp = agent[j][i]
                
                if rs == True:
                    frame = resize(frame, (105, 80))
                    comp = resize(comp, (105, 80))
                
                e = entropy(frame.reshape(-1), comp.reshape(-1))
                if e < 1000:
                    all_entropies.append(e)
    
    print('Entropy over %i frames' % len(all_entropies))
    print('Mean entropy: ', np.mean(all_entropies))
    print('Standard dev: ', np.std(all_entropies))
    
    return all_entropies




# Calculate KL-divergence over randomly paired human-agent frames
# include rs=True for resizing to original size

def get_randent(human, agent, rs=False):    
    all_rand_entropies = []
    for j in range(len(human)):
        for i in range(len(human[j])):
            frame = human[j][i]
            indices = np.where(frame==1)
            
            if indices[0].any():
                frame = frame * 100
                frame = cv2.GaussianBlur(frame,(31,31),8)
                
                first_ind = randint(len(agent), size=1)
                second_ind = randint(len(agent[first_ind[0]]), size=1)
                
                comp = agent[first_ind[0]][second_ind[0]]
                
                if rs == True:
                    frame = resize(frame, (105, 80))
                    comp = resize(comp, (105, 80))
                
                e = entropy(frame.reshape(-1), comp.reshape(-1))
                
                if e < 1000:
                    all_rand_entropies.append(e)

    print('Random entropy over %i frames' % len(all_rand_entropies))
    print('Mean entropy: ', np.mean(all_rand_entropies))
    print('Standard dev: ', np.std(all_rand_entropies))
    
    return all_rand_entropies


# Calculate KL-divergence for human-human paired (watching only) data
def get_hum_ent(human):
    
    all_entropies = []
    
    human1 = human[4]
    for j in range(len(human)):
        for i in range(len(human1)):
            frame1 = human1[i]
            frame2 = human[j][i]
            indices1 = np.where(frame1==1)
            indices2 = np.where(frame2==1)
            
            if indices1[0].any() and indices2[0].any():
                frame1 = frame1 * 100
                frame1 = cv2.GaussianBlur(frame1,(31,31),8)
                
                frame2 = frame2 * 100
                frame2 = cv2.GaussianBlur(frame2,(31,31),8)
                
                e = entropy(frame1.reshape(-1), frame2.reshape(-1))
                
                if e < 1000000:
                    all_entropies.append(e)
    
    print('Entropy over %i frames' % len(all_entropies))
    print('Mean entropy: ', np.mean(all_entropies))
    print('Standard dev: ', np.std(all_entropies))
    
    return all_entropies


# Calculate KL-divergence for randomly paired human-human (watching only) data
def get_hum_randent(human):
    
    all_entropies = []
    
    human1 = human[4]
    for j in range(len(human)):
        for i in range(len(human1)):
            frame1 = human1[i]
            frame2 = human[j][randint(len(human[j]), size=1)[0]]
            indices1 = np.where(frame1==1)
            indices2 = np.where(frame2==1)
            
            if indices1[0].any() and indices2[0].any():
                frame1 = frame1 * 100
                frame1 = cv2.GaussianBlur(frame1,(31,31),8)
                
                frame2 = frame2 * 100
                frame2 = cv2.GaussianBlur(frame2,(31,31),8)
                
                e = entropy(frame1.reshape(-1), frame2.reshape(-1))
                
                if e < 1000000:
                    all_entropies.append(e)
    
    print('Random entropy over %i frames' % len(all_entropies))
    print('Mean entropy: ', np.mean(all_entropies))
    print('Standard dev: ', np.std(all_entropies))
    
    return all_entropies


print('Start KL-divergence calculations: \n\n')


print('-- Human-human comparison --')
hw_one = get_hum_ent(wh)
print('-- Randomized human-human comparison --')
humsrand = get_hum_randent(wh)

humtest = ttest_ind(hw_one, humsrand)
print(' ')
print('P-value for difference between comparisons: ', humtest[1])
print(' ')
print(' ')


print('-- Human-agent comparison (Playing) --')
hpa_one = get_ent(ph, pa)
print('-- Randomized human-agent comparison --')
hpa_randone = get_randent(ph, pa)

hpa_test = ttest_ind(hpa_one, hpa_randone)
print(' ')
print('P-value for difference between comparisons: ', hpa_test[1])
print(' ')
print(' ')




print('-- Human-agent comparison (Watching) --')
hwa_one = get_ent(wh, wa)
print('-- Randomized human-agent comparison --')
hwa_randone = get_randent(wh, wa)

hwa_test = ttest_ind(hwa_one, hwa_randone)
print(' ')
print('P-value for difference between comparisons: ', hwa_test[1])
print(' ')
print(' ')


print('Statistical script done.')
