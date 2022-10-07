'''This file enables straightforward reproduction of some important experiments:
training models on the training set only and evaluating them on the test
set, building the final model, scoring all sequences and extracting the 
top 500, scoring the wild-type.'''

#Author: Jonathan Parkinson <jlparkinson1@gmail.com>

import os
import sys
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from ..utilities.model_data_loader import load_model, save_model, load_data
from torch.utils.data import random_split
from sklearn.model_selection import KFold
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.ensemble import RandomForestClassifier
from ..model_code.traditional_nn_classification import fcnn_classifier as FCNN
from ..model_code.variational_Bayes_ordinal_reg import bayes_ordinal_nn as BON

#TODO: Move full_wt_seq and aas to a constants file.
full_wt_seq = ('EVQLVESGGGLVQPGGSLRLSCAASGFTFSD--SWIHWVRQAPGKGLEWVAWISP--'
                'YGGSTYYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCARRHWPGGF----------DYWGQGTLVTVSS')

aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
       'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']




def get_wt_score(start_dir):
    """This function scores the wild-type sequence to see how it compares."""
    autoencoder = load_model(start_dir, "TaskAdapted_Autoencoder.ptc",
            model_type="adapted")
    varbayes_model = load_model(start_dir, "atezolizumab_varbayes_model.ptc",
            model_type="BON")
    if autoencoder is None:
        raise ValueError("Autoencoder not yet trained / saved.")
    
    onehot_wt = np.zeros((1,132,21))
    for i in range(len(full_wt_seq)):
        onehot_wt[0,i,aas.index(full_wt_seq[i])] = 1.0
    onehot_wt = torch.from_numpy(onehot_wt).float()
    x = autoencoder.extract_hidden_rep(onehot_wt.cuda()).cpu()
    x = x[:,29:124,:].flatten(1,-1)
    score, stdev = varbayes_model.extract_hidden_rep(x, use_MAP=True)
    print("WT MAP score: %s"%score.numpy().flatten()[0])

def get_best_seqs(start_dir, num_to_keep = 4000):
    """This function scores all sequences in both training and test sets,
    then extracts the best 500 of those it finds (using MAP predictions
    for reproducibility).

    Args:
        start_dir (str): A path to the project dir.
        num_to_keep (int): The number of sequences to keep.
    """
    autoencoder = load_model(start_dir, "TaskAdapted_Autoencoder.ptc",
            model_type="adapted")
    varbayes_model = load_model(start_dir, "atezolizumab_varbayes_model.ptc",
            model_type = "BON")
    if autoencoder is None or varbayes_model is None:
        raise ValueError("Autoencoder not yet trained / saved, or "
            "final atezolizumab model not yet trained / saved.")
    rawx1, rawy1, rawx2, rawy2 = load_data(start_dir, data_type="onehot")
    if rawx1 is None:
        raise ValueError("Encoded data has not yet been generated.")
    y = torch.cat([rawy1, rawy2], dim=0)
    x = torch.cat([rawx1, rawx2], dim=0)
    
    x_encoded = autoencoder.extract_hidden_rep(x.cuda())
    x_encoded = x_encoded[:,29:124,:].flatten(1,-1)
    xscores, stdevs = varbayes_model.extract_hidden_rep(x_encoded, use_MAP=True)
    xscores = xscores.numpy().flatten()
    top_indices = np.argsort(xscores)[-num_to_keep:]
    top_indices = torch.from_numpy(top_indices)
    best_onehot = x[top_indices,:,:]
    best_scores = xscores[top_indices]
    
    os.chdir(start_dir)
    os.chdir(os.path.join("results_and_resources", "selected_sequences"))
    torch.save(best_onehot, "top_%s_sequences.ptc"%num_to_keep)
    textfile_copy = open("top_%s_sequences.txt"%num_to_keep, "w+")
    #The next part is slightly awkward. Because the sequences were one-hot
    #encoded THEN train-test split, then the one-hot was used to generate
    #the other encodings...we have to convert the one-hot back into sequence.
    #We could have avoided this by assigning each sequence a unique ID, or
    #by doing the train-test split before the onehot. At this point however
    #this is baked into the pipeline and for now is only a minor inconvenience 
    #for now...
    for i in range(best_onehot.size()[0]):
        current_seq = []
        for j in range(best_onehot.size()[1]):
            current_seq.append(aas[torch.argmax(best_onehot[i,j,:]).item()])
        current_seq.append(" %s"%best_scores[i])
        current_seq.append("\n")
        textfile_copy.write(''.join(current_seq))
    textfile_copy.close()
    os.chdir(start_dir)


def plot_all_scores(start_dir):
    """This function scores all sequences in both training and test sets,
    then plots the scores of those it finds (using MAP for reproducibility).

    Args:
        start_dir (str): A path to the project dir.
        num_to_keep (int): The number of sequences to keep.
    """
    autoencoder = load_model(start_dir, "TaskAdapted_Autoencoder.ptc",
            model_type="adapted")
    varbayes_model = load_model(start_dir, "atezolizumab_varbayes_model.ptc",
            model_type = "BON")
    if autoencoder is None or varbayes_model is None:
        raise ValueError("Autoencoder not yet trained / saved, or "
            "final atezolizumab model not yet trained / saved.")
    rawx1, rawy1, rawx2, rawy2 = load_data(start_dir, data_type="onehot")
    if rawx1 is None:
        raise ValueError("Encoded data has not yet been generated.")
    y = torch.cat([rawy1, rawy2], dim=0).numpy()
    x = torch.cat([rawx1, rawx2], dim=0)
    label_key = {0:"RH01", 1:"RH02", 2:"RH03"}
    labels = [label_key[category] for category in y[:,-2].tolist()]

    x_encoded = autoencoder.extract_hidden_rep(x.cuda())
    x_encoded = x_encoded[:,29:124,:].flatten(1,-1)
    xscores, stdevs = varbayes_model.extract_hidden_rep(x_encoded, use_MAP=True)
    xscores = xscores.numpy().flatten()
    
    os.chdir(start_dir)
    os.chdir(os.path.join("results_and_resources", "selected_sequences"))
    plt.style.use("bmh")
    fig, ax = plt.subplots(1, figsize=(8,6))
    sns.kdeplot(x = xscores, hue = labels, ax = ax)
    ax.set_xlabel("Score assigned by variational\nBayesian ordinal regression")
    ax.set_ylabel("PDF")
    plt.title("Distribution of scores assigned by variational Bayesian\n"
            "ordinal regression vs experimentally determined\n"
            "binding category")
    plt.savefig("Score_distribution.png")
    plt.close()
    os.chdir(start_dir)



def train_final_model(start_dir, num_epochs):
    """This function trains and saves the final model on the entire dataset.
    The utilities functions will not let you overwrite the final saved model
     -- you can manually delete it and then retrain it if you like.
    Use a random seed for reproducibility."""
    trainx, trainy, testx, testy = load_data(start_dir, "adapted")
    xtrain = torch.cat([trainx, testx], dim=0)
    ytrain = torch.cat([trainy, testy], dim=0)
    xtrain = xtrain.flatten(1,-1)
    
    model = BON(input_dim=xtrain.size()[1])
    losses = model.trainmod(xtrain, ytrain, lr=0.005, scale_data = True,
            num_samples = 10)
    save_model(start_dir, "atezolizumab_varbayes_model.ptc", model) 

def get_class_weights(y, as_dict = False):
    """This function is used to generate class weights for the random forest and
    fully connected classifiers, which are generated as baselines for comparison
    with the variational network. TODO: Merge with dup fun in cv_scoring.

    Args:
        y (tensor): The y data. Should be an N x 4 array. The first two columns
            are class labels encoded for ordinal regression. Last two are
            class label as integer and datapoint weight.
        as_dict (bool): If True, return the weights as a dictionary (for random
            forest model).

    Returns:
        class_weights: Either a tensor (as_dict = False) or a dict.
    """
    classes = y[:,-2].numpy()
    n_instances = np.asarray([classes.shape[0] / np.argwhere(classes==0).shape[0],
                                classes.shape[0] / np.argwhere(classes==1).shape[0],
                                classes.shape[0] / np.argwhere(classes==2).shape[0]])
    n_instances = n_instances / np.max(n_instances)
    if as_dict == False:
        return torch.from_numpy(n_instances)
    return {0:n_instances[0], 1:n_instances[1], 2:n_instances[2]}


def eval_train_test(start_dir, data_type, num_epochs, model_type):
    """This function builds the specified model type on the specified encoding
    type using the traiing set only if that model does not already exist
    (if the model does already exist it is loaded). Next, it generates
    predictions for the test set and scores them for accuracy then returns
    the scores. Use a random seed for reproducibility.

    Args:
        start_dir (str): The project directory filepath.
        data_type (str): The type of encoding to use.
        num_epochs (int): The number of epochs to train for.
        model_type (str): The model type (fully connected, variational etc.)
    """
    trainx, trainy, testx, testy = load_data(start_dir, data_type)
    if trainx is None:
        raise ValueError("The data type selected by model_training_code "
                "has not yet been generated.")
    #The autoencoder-generated data is flattened from a 3d tensor to a matrix.
    #All other data types are already "flat".
    if len(trainx.size()) ==3:
        trainx = trainx.flatten(1,-1)
        testx = testx.flatten(1,-1)
    #The FCNN class does not auto-scale the data unlike BON so we do that here.
    #Very important obviously that we not try to scale onehot data...
    if model_type == "FCNN" and data_type != "onehot":
        testx = (testx - torch.mean(trainx, dim=0).unsqueeze(0)) / torch.std(trainx, dim=0).unsqueeze(0)
        trainx = (trainx - torch.mean(trainx, dim=0).unsqueeze(0)) / torch.std(trainx, dim=0).unsqueeze(0)

    model = None
    print("*******\nNOW EVALUATING %s using %s"%(data_type, model_type))
    if model is None:
        #Very important not to try to scale onehot data...
        scale_data = True
        if data_type == "onehot":
            scale_data = False
        print("Model for data type %s not found...will train."%data_type)
        if model_type == "BON":
            model = BON(input_dim=trainx.size()[1], num_categories = trainy.shape[1] - 1)
            random_seed = 0
            #For some reason, the fair_esm data encoding type gave HORRIBLE
            #results using the same random seed as all the others. This is 
            #definitely a black mark against fair_esm...but just to give
            #fair_esm a "fair shake", we tried using a different random seed
            #for it just in case. It still didn't perform well...
            if data_type == "fair_esm":
                random_seed = 10
            losses = model.trainmod(trainx, trainy, 
                    epochs=num_epochs, scale_data=scale_data, random_seed=0,
                    num_samples=10)
            save_model(start_dir, "%s_trainset_only_%s_model.ptc"%(data_type, model_type),
                                model)
        elif model_type == "FCNN":
            class_weights = get_class_weights(trainy)
            model = FCNN(dropout = 0.2, input_dim = trainx.size()[1])
            losses = model.trainmod(trainx, trainy, epochs=num_epochs, class_weights = 
                    class_weights, lr=0.005)
            save_model(start_dir, "%s_trainset_only_%s_model.ptc"%(data_type, model_type),
                                model)
        elif model_type == "RF":
            class_weights = get_class_weights(trainy, as_dict=True)
            model = RandomForestClassifier(n_estimators=500,
                        min_samples_split=25, min_samples_leaf=4, 
                        n_jobs=3, class_weight=class_weights,
                        random_state = 0)
            model.fit(trainx, trainy[:,-2], sample_weight=trainy[:,-1])
            save_model(start_dir, "%s_trainset_only_RF_model.pk"%(data_type),
                                model)
    if model_type == "BON":
        trainpreds = model.map_categorize(trainx)
        testpreds = model.map_categorize(testx)
    elif model_type == "FCNN":
        trainpreds = model.predict(trainx)[1]
        testpreds = model.predict(testx)[1]
    elif model_type == "RF":
        trainpreds = model.predict(trainx)
        testpreds = model.predict(testx)
    else:
        raise ValueError("Unrecognized model type passed to model_training_code.")
    testscore = mcc(testy[:,-2], testpreds)
    trainscore = mcc(trainy[:,-2], trainpreds)
    print("Trainscore: %s"%trainscore)
    print("Testscore: %s"%testscore)
    return trainscore, testscore


def train_evaluate_models(start_dir, action_to_take):
    """Convenience function for training models on the training set, evaluating them on the test set,
    building a final model, scoring the wild type and scoring the sequences in the original dataset,
    then selecting the best.

    Args:
        start_dir (str): A filepath to the project directory.
        action_to_take (str): One of 'traintest_eval', 'train_final_model'.
    """
    os.chdir(os.path.join(start_dir, "encoded_data"))
    fnames = os.listdir()
    #This is a little...clunky, but because of the way the pipeline is set up, we have to encode the
    #data using a variety of different schemes and should make sure all are present before proceeding.
    for expected_fname in ["adapted_testx.pt", "adapted_trainx.pt", "nonadapted_testx.pt",
            "nonadapted_trainx.pt", "onehot_testx.pt", "onehot_trainx.pt", "protvec_testx.pt",
            "protvec_trainx.pt", "testy.pt", "trainy.pt", "unirep_testx.pt", "unirep_testy.pt",
            "unirep_trainx.pt", "unirep_trainy.pt", "fair_esm_trainx.pt", "fair_esm_testx.pt"]:
        if expected_fname not in fnames:
            print("The data has not been encoded using all of the expected encoding types. This "
                    "step is required before train test evaluation.")
            return
    if action_to_take == "traintest_eval":
        test_results_dict = dict()
        for data_type in ["adapted", "nonadapted", "onehot",
                "protvec", "unirep", "fair_esm"]:
            test_results_dict[data_type + "_BON"] = eval_train_test(start_dir, data_type, 
                    num_epochs=40, model_type = "BON")
        for data_type in ["adapted", "onehot", "unirep", "fair_esm", "protvec"]:
            test_results_dict[data_type + "FCNN"] = eval_train_test(start_dir, data_type, 
                    num_epochs=40, model_type = "FCNN")
        test_results_dict["adapted_RF"] = eval_train_test(start_dir, "adapted", 
                    num_epochs=0, model_type = "RF")
        test_results_dict["onehot_RF"] = eval_train_test(start_dir, "onehot", 
                    num_epochs=0, model_type = "RF")
        os.chdir(start_dir)
        os.chdir("results_and_resources")
        with open("test_set_results.txt", "w+") as outf:
            outf.write("Data type_model,MCC train,MCC test\n")
            for key in test_results_dict:
                outf.write(key)
                outf.write(",")
                outf.write(",".join([str(k) for k in test_results_dict[key]]))
                outf.write("\n")
    elif action_to_take == "train_final_model":
        os.chdir(os.path.join(start_dir, "results_and_resources", "trained_models"))
        if "atezolizumab_varbayes_model.ptc" in os.listdir():
            print("A final model has already been trained and is saved under "
                    "'results_and_resources/trained_models'. To guarantee "
                    "reproducibility, retraining the final model is not "
                    "encouraged. If you wish to do so, you can delete the final "
                    "model then retrain, but again, this is not a recommended step.")
        return
        train_final_model(start_dir, num_epochs=30)
    elif action_to_take == "score_wt":
        get_wt_score(start_dir)
    elif action_to_take == "get_top_seqs":
        get_best_seqs(start_dir)
        print("Top sequences retrieved, saved to 'results_and_resources/selected_sequences'.")
    else:
        raise ValueError("Invalid action specified when calling model_training_code.")


def train_evaluate_trastuzumab(project_dir):
    """Convenience function for training models on the training set and evaluating on the
    test set for trastuzumab specifically.

    Args:
        project_dir (str): A filepath to the project directory.
    """
    os.chdir(os.path.join(project_dir, "trastuzumab_data"))
    start_dir = os.getcwd()
    os.chdir("encoded_data")
    fnames = os.listdir()
    #This is a little...clunky, but because of the way the pipeline is set up, we have to encode the
    #data using a variety of different schemes and should make sure all are present before proceeding.
    for expected_fname in ["adapted_testx.pt", "adapted_trainx.pt"]:
        if expected_fname not in fnames:
            print("The data has not been encoded using all of the expected encoding types. This "
                    "step is required before train test evaluation.")
            return
    test_results = eval_train_test(start_dir, "adapted", 
                    num_epochs=40, model_type = "BON")
    os.chdir(project_dir)
    os.chdir("results_and_resources")
    with open("trastuzumab.txt", "w+") as outf:
        outf.write("Data type_model,MCC train,MCC test\n")
        outf.write(key)
        outf.write(",")
        outf.write(",".join([str(k) for k in test_results]))
        outf.write("\n")
