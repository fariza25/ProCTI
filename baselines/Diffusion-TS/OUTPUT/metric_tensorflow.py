import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
sys.path.append(os.path.join(os.path.dirname('__file__'), '../'))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from Utils.metric_utils import display_scores
from Utils.discriminative_metric import discriminative_score_metrics
from Utils.predictive_metric import predictive_score_metrics

iterations = 5
ori_data = np.load('energy/samples/energy_norm_truth_100_train.npy')
# ori_data = np.load('../OUTPUT/{dataset_name}/samples/{dataset_name}_norm_truth_{seq_length}_train.npy')  # Uncomment the line if dataset other than Sine is used.
fake_data = np.load('energy/ddpm_fake_energy.npy')

discriminative_score = []

for i in range(iterations):
    temp_disc, fake_acc, real_acc = discriminative_score_metrics(ori_data[:], fake_data[:ori_data.shape[0]])
    discriminative_score.append(temp_disc)
    print(f'Iter {i}: ', temp_disc, ',', fake_acc, ',', real_acc, '\n')
      
print('Discriminative')
display_scores(discriminative_score)
print()

predictive_score = []
for i in range(iterations):
    temp_pred = predictive_score_metrics(ori_data, fake_data[:ori_data.shape[0]])
    predictive_score.append(temp_pred)
    print(i, ' epoch: ', temp_pred, '\n')
      
print('Predictive')
display_scores(predictive_score)
print()
