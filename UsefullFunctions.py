import sys
sys.path.append('../')

import numpy as np
from sklearn import svm
from sklearn.linear_model import Perceptron

import matplotlib.pyplot as plt
import matplotlib 
import pandas as pd
import seaborn as sbn
from contextlib import redirect_stdout
import json

import Reinforce as rln


def frates_labels(iterations):
    
    reinforce = rln.REINFORCE(name_load_actor="models/RL_actor_network_good.pt",
                              name_load_critic="models/RL_critic_network_good.pt")
    
    observations, rewards, actions,\
    log_action_probs, entropies, values,\
    trial_begins, errors, frates_actor, frates_critic,\
    timeav_values, final_actions, global_values, stimuli = reinforce.experience(iterations)
    
    for i in range (len(global_values)):
        if global_values[i] <= 1:
            global_values[i] = -1
        else:
            global_values[i] = 1
            
    array1_list = frates_actor[:, 1:].T.tolist()
    array2_list = frates_critic[:, 1:].T.tolist()
    array3_list = final_actions[1:].tolist()
    array4_list = global_values[1:].tolist()
    array5_list = stimuli[1:].tolist()
    
    data = {
        "frates_actor": array1_list,
        "frates_critic": array2_list,
        "final_actions": array3_list,
        "global_values": array4_list,    
        "stimuli": array5_list
    }
    
    with open('frates_labels.json', 'w') as json_file:
        json.dump(data, json_file)

    





def rel_nurons(X, Y, model, C, network, label):
    
    model = model
        
    if model == 'perceptronL1':
        C_perc = C
    elif model == 'svm':
        C_svm = C
            
    nb_epochs = 10

    test_scores = np.zeros(nb_epochs)
    
    nb_trials = X.shape[0]
    percentage_training_set = 0.8
    nb_indeces_training = int(nb_trials*percentage_training_set)
    
    for i in range(nb_epochs):
        
        all_indeces = np.arange(0, nb_trials)
        
        if i == 0:
            indeces_train = all_indeces[0:nb_indeces_training]
            indeces_test = all_indeces[nb_indeces_training:]
        else:
            np.random.shuffle(all_indeces)
            indeces_train = all_indeces[0:nb_indeces_training]
            indeces_test = all_indeces[nb_indeces_training:]
    
        X_train_trial = X[indeces_train,:]
        Y_train_trial = Y[indeces_train]
        X_test_trial = X[indeces_test,:]
        Y_test_trial = Y[indeces_test]
        
        if model=='perceptron':
            clf = Perceptron(tol=1e-3, random_state=0)
        elif model == 'perceptronL1':
            clf = Perceptron(tol=1e-3, random_state=0, penalty='l1', alpha=C_perc)
        elif model == 'svm':
            clf = svm.LinearSVC(penalty='l1', C=C_svm, dual = False, max_iter=1000)
    
        clf.fit(X_train_trial, Y_train_trial)
        training_score = clf.score(X_train_trial, Y_train_trial)
        test_score = clf.score(X_test_trial, Y_test_trial)
    
        test_scores[i] = test_score
    
    #----------------------------------------------------------------------------------------#

    test_random_scores = np.zeros(nb_epochs)

    for i in range(nb_epochs):
        
        all_indeces = np.arange(0, nb_trials)
        
        if i == 0:
            indeces_train = all_indeces[0:nb_indeces_training]
            indeces_test = all_indeces[nb_indeces_training:]
        else:
            np.random.shuffle(all_indeces)
            indeces_train = all_indeces[0:nb_indeces_training]
            indeces_test = all_indeces[nb_indeces_training:]
    
        X_train_trial = X[indeces_train,:]
        Y_train_trial = Y[indeces_train]
        X_test_trial = X[indeces_test,:]
        #Y_test_trial = Y[indeces_test]
        #Y_train_trial = 2*np.random.binomial(size=497, n=1, p=0.5)-1
        Y_test_trial = 2*np.random.binomial(size=200, n=1, p=0.5)-1 ##before size was 40?!?
        
        if model=='perceptron':
            clf = Perceptron(tol=1e-3, random_state=0)
        elif model == 'perceptronL1':
            clf == Perceptron(tol=1e-3, random_state=0, penalty='l1', alpha=C_perc)
        elif model == 'svm':
            clf = svm.LinearSVC(penalty='l1', C=C_svm, dual=False, max_iter=1000)
        
        clf.fit(X_train_trial, Y_train_trial)
        training_score = clf.score(X_train_trial, Y_train_trial)
        test_score = clf.score(X_test_trial, Y_test_trial)
        
        test_random_scores[i] = test_score
                
    plt.figure(figsize=(6,4))
    plt.hist(test_scores, label="test scores")
    plt.hist(test_random_scores, label="test random scores")
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.legend(fontsize=15, loc="upper left")
    plt.title(network+" network / "+label+ " test scores", fontsize=20);
    plt.savefig('clf_data/'+label+'/hist: '+network+' - '+label+'.png')
    
    print("average over 10 epochs of test scores: ", np.mean(test_scores))
    print("average over 10 epochs of test random scores: ", np.mean(test_random_scores))   

    #----------------------------------------------------------------------------------------#

    all_indeces = np.arange(0, nb_trials)
    np.random.shuffle(all_indeces)
    indeces_train = all_indeces[0:nb_indeces_training]
    indeces_test = all_indeces[nb_indeces_training:]
    X_train_trial = X[indeces_train,:]
    Y_train_trial = Y[indeces_train]
    X_test_trial = X[indeces_test,:]
    Y_test_trial = Y[indeces_test]
    
    if model=='perceptron':
        clf = Perceptron(tol=1e-3, random_state=0)
    elif model == 'perceptronL1':
        clf == Perceptron(tol=1e-3, random_state=0, penalty='l1', alpha=C_perc)
    elif model == 'svm':
        clf = svm.LinearSVC(penalty='l1', C=C_svm, dual=False, max_iter=1000)
        
    clf.fit(X_train_trial, Y_train_trial)
    training_score = clf.score(X_train_trial, Y_train_trial)
    test_score = clf.score(X_test_trial, Y_test_trial)
    print("----------\ntraining score: ", training_score)
    print("test score: ", test_score, "\n----------")
    
    w = clf.coef_
    b = clf.intercept_
    
    df = pd.DataFrame(w[0,:])
    df.to_csv("clf_data/"+label+"/perceptron_wo_"+network+".csv", index=False)
     
    relevant_neurons = []
    relevant_neurons_values = []
    plt.figure(figsize=(15,7))
    plt.plot(w[0,:])
    for i in range(len(w[0,:])):
        if w[0,i] != 0:
            plt.text(i, w[0,i], str(i), style='italic', fontsize=15)
            relevant_neurons_values.append(np.abs(w[0,i]))
            relevant_neurons.append(i)
    plt.title(network+" network relevant neurons for "+label, fontsize=20);
    plt.xticks(size=15)
    plt.yticks(size=15)
    
    sorted_pairs = sorted(zip(relevant_neurons_values, relevant_neurons))
    relevant_neurons = [pair[1] for pair in sorted_pairs]
    relevant_neurons.reverse()
    relevant_size = 10
    relevant_neurons = relevant_neurons[:relevant_size]
    
    if network == "actor":
        with open('clf_data/'+label+'/relevant_neurons_actor.txt', 'w') as f:
            with redirect_stdout(f):
                print(relevant_neurons)
    else:
        with open('clf_data/'+label+'/relevant_neurons_critic.txt', 'w') as f:
            with redirect_stdout(f):
                print(relevant_neurons)    
            
            
            
            
            
def tuning_curves(relevant_neurons, X, stimuli, network, label):
    
    frates = X.T   
    frates_rid = frates[relevant_neurons]
    #rates_rid = frates_rid.mean(axis=1)
    x_values = np.zeros(stimuli.shape[0])
    
    if label == "actions":
        for i in range(len(x_values)):
            x_values[i] = stimuli[i][0]*stimuli[i][1] - stimuli[i][2]*stimuli[i][3]
            
    fig, axx = plt.subplots(5, 2, figsize=(15, 20))
    axx = axx.reshape(-1)
    for n, ax in enumerate(axx):
        for i in range(len(relevant_neurons)):
            ax.plot(x_values[i], frates_rid[n, i], "*", markersize=10, color="purple")
            #ax.text(x_values[i]+0.03, collection1[n, i], pair[i], fontsize=10)
            ax.set_title("neuron %i" %(relevant_neurons[n]), size=20)
            ax.tick_params(axis='x', labelsize=15)
            ax.tick_params(axis='y', labelsize=15)
            ax.set_xlabel("v1p1", size=15)
            ax.set_ylabel("firing rate", size=15)
    plt.tight_layout()      
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    #new_vector[:,0] = a_frates_rid
    #        collection1 = np.concatenate((collection1, new_vector), axis=1)
            
    #collection1 = collection1[:, 1:]
    