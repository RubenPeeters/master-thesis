import os
import gym
import glob
import time
import json
import random
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

from stable_baselines import A2C
from stable_baselines import DQN

import tensorflow as tf

datasets = ['nslkdd', 'unswnb15', 'cicddos2019', 'cicddos2019-top', 'cicdos2017']

def mal_ben(df):
    print(df['class'].value_counts())
    df['class'] = df['class'].astype('object')
    atk_idx = df.loc[df['class'] != "normal"].index
    df.loc[atk_idx, 'class'] = 1.0
    df.loc[df.index.difference(atk_idx), 'class'] = 0.0
    df['class'] = df['class'].astype(dtype=np.float32)



def nslkdd_split_df(df):
    col = df.columns[-1]
    cols = df.columns[:-1]

    return df[train_cols], df[train_col]

def ids_eval(model):
    TP, FP, TN, FN = 0,0,0,0
    env = IdsEnv(images_per_episode=1, dataset=(x_test, y_test), random=False)
    obs, done = env.reset(), False
    try:
        while True:
            obs, done = env.reset(), False
            while not done:
                obs, rew, done, info = env.step(model.predict(obs)[0])
                label = info['label']
                if label == 0 and rew > 0:
                    TP += 1
                if label == 0 and rew == 0:
                    FP += 1
                if label == 1 and rew > 0:
                    TN += 1
                if label == 1 and rew == 0:
                    FN += 1

    except StopIteration:
        accuracy = (float(TP + TN) / (TP + FP + FN + TN)) 
        precision = (float(TP) / (TP + FP))
        recall = (float(TP) / (TP + FN)) # = TPR = Sensitivity
        FPR = (float(FP) / (TN + FP)) # 1 - specificity
        f1_score = 2 * (precision * recall) / (precision + recall)
        print()
        print('validation done...')
        print('Accuracy: {0}%'.format(accuracy * 100))
        print('Precision: {0}%'.format(precision * 100))
        print('Recall/TPR/Sensitivity: {0}%'.format(recall * 100))
        print('FPR: {0}%'.format(FPR * 100))
        print('F1 score: {0}'.format(f1_score))
    return [accuracy, precision, recall, FPR, f1_score]


end_d = {'model', 'train dataset', 'test dataset', 'features', 'accuracy', 'precision', 'recall', 'FPR', 'F1 score', 'notes'}
end_df = pd.DataFrame(data=end_d)

for d in datasets:
    dd = './models/{d}/DQN'
    ad = './models/{d}/A2C'
    os.chdir(dd)
    for m in glob.glob("*.pkl"):
        print(m)
        model = DQN.load(f'{m}')
        if d is 'nslkdd':
            df_train = pd.read_feather('/project/datasets/clean-ids-collection/nsl-kdd/clean/KDDTrain.feather')
            df_test = pd.read_feather('/project/datasets/clean-ids-collection/nsl-kdd/clean/KDDTest.feather')
            df = pd.concat([df_train, df_test], ignore_index=True)
            x_test, y_test = nslkdd_split_df(df)
            results = ids_eval(model)
            entry = ['{m}', 'nslkdd', 'nslkdd', '{df.columns}'] + results
            end_df.append(entry)
        if d is 'unswnb15':
            df_train = pd.read_feather('/project/datasets/clean-ids-collection/nsl-kdd/clean/KDDTrain.feather')
            df_test = pd.read_feather('/project/datasets/clean-ids-collection/nsl-kdd/clean/KDDTrain.feather')
            df = pd.concat([df_train, df_test], ignore_index=True)
            x_test, y_test = nslkdd_split_df(df)
            results = ids_eval(model)
            entry = ['{m}', 'unswnb15', 'unswnb15', '{df.columns}'] + results
            end_df.append(entry)
