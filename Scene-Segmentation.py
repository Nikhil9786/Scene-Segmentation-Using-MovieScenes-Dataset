# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 20:09:16 2021

@author: Nikhil Arora
"""

import os
import pandas as pd
import glob
import pickle
import torch
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def make_predictions(data_dir, res_dir):
    '''
    Function reads all pkl files, concate them into one and converts it 
    into Dataframe to predict probabilities of a shot boundary for it to 
    be a scene boundary

    Parameters
    ----------
    data_dir : TYPE
        DESCRIPTION: pkl files 
    res_dir : TYPE
        DESCRIPTION pkl files saved with predicted probabilities

    Returns
    -------
    None.

    '''
    filenames = glob.glob(os.path.join(data_dir, "tt*.pkl"))
    
    for fn in filenames:
    
        x = pickle.load(open(fn, "rb"))

        #Declaring dictionary for every key in pickle file
        place_dict = dict()
        cast_dict = dict()
        action_dict = dict()
        audio_dict = dict()
        gt_dict = dict()
        pr_dict = dict()
        shot_to_end_frame_dict = dict()

        place_dict[x["imdb_id"]] = x["place"].numpy()
        cast_dict[x["imdb_id"]] = x["cast"].numpy()
        action_dict[x["imdb_id"]] = x["action"].numpy()
        audio_dict[x["imdb_id"]] = x["audio"].numpy()
        gt_dict[x["imdb_id"]] = x["scene_transition_boundary_ground_truth"].numpy()
        pr_dict[x["imdb_id"]] = x["scene_transition_boundary_prediction"].numpy()
        shot_to_end_frame_dict[x["imdb_id"]] = x["shot_end_frame"].numpy()

        df1 = pd.DataFrame.from_dict(place_dict[x["imdb_id"]])
        df2 = pd.DataFrame.from_dict(cast_dict[x["imdb_id"]])
        df3 = pd.DataFrame.from_dict(action_dict[x["imdb_id"]])
        df4 = pd.DataFrame.from_dict(audio_dict[x["imdb_id"]])

        df = pd.concat([df1, df2, df3, df4], axis=1)

        df["scene_transition_boundary_ground_truth"] = pd.DataFrame.from_dict(gt_dict[x["imdb_id"]])

        # Dividing the dataframe into Features and Predictors
        X = df[:len(place_dict[x["imdb_id"]])-1].drop(['scene_transition_boundary_ground_truth'],axis=1)
        Y = df[:len(place_dict[x["imdb_id"]])-1]["scene_transition_boundary_ground_truth"].astype(int)

        # Spliting the dataset into Training and Testing into 80% and 20% respectively
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 42)
        
        
        '''
        Comparing different models for best prediction model
        '''
        
        # model = LinearRegression().fit(X_train, Y_train)
        # model = RandomForestClassifier(max_depth=2, random_state = 42).fit(X_train,Y_train)
        # model = SGDClassifier(max_iter = 2000, tol = 1e-3).fit(X_train,Y_train)
        # model = GaussianNB().fit(X_train,Y_train)
        # model = KNeighborsClassifier(n_neighbors=3).fit(X_train,Y_train)
        model = LogisticRegression(random_state=0,max_iter=3000).fit(X_train, Y_train)

        print("Accuracy for IMDB " + x["imdb_id"] + ": " +str(model.score(X_test, Y_test)))

        pr = model.predict_proba(X)

        '''
        Selecting the second element as it tells the probability of the scene transition boundary.
        '''
        
        pr_ls = []
        for i in range(len(pr)):
            pr_ls.append(pr[i][1])

        
        # Converting the Numpy into Tensor
        x['scene_transition_boundary_prediction'] = torch.FloatTensor(pr_ls) 
        
        ans_file = res_dir + "/" + x["imdb_id"] + ".pkl"

        # Dumping the results in pkl files
        pickle.dump( x, open( ans_file, "wb" ) )

if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1]
    res_dir = sys.argv[2]

    make_predictions(data_dir, res_dir)