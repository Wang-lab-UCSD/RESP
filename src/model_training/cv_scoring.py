'''This file enables reproduction of cross-validation. It's not part of the
standard pipeline because it's computationally somewhat more expensive --
a model is trained and scored using 5x CV on each encoding of the atezolizumab
dataset -- but it's easy to run if desired. Just use the run_cv script in the
main directory.

Note that the 5x CV is run using
the training set only. When the pipeline was originally developed, all model
selection was done using this CV procedure; the train-test split was only
performed once all the encoding schemes had been developed and model
hyperparameters had been selected.'''

#Author: Jonathan Parkinson <jlparkinson1@gmail.com>

import os, sys, numpy as np, torch
from ..utilities.model_data_loader import load_data
from torch.utils.data import random_split
from sklearn.model_selection import KFold
from sklearn.metrics import matthews_corrcoef as mcc, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from ..model_code.traditional_nn_classification import fcnn_classifier as FCNN
from ..model_code.variational_Bayes_ordinal_reg import bayes_ordinal_nn as BON
from sklearn.model_selection import KFold
import time

full_wt_seq = ('EVQLVESGGGLVQPGGSLRLSCAASGFTFSD--SWIHWVRQAPGKGLEWVAWISP--'
                'YGGSTYYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCARRHWPGGF----------DYWGQGTLVTVSS')

aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
       'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']


#This function is used to generate class weights for the random forest and
#fully connected classifiers, which are generated as baselines for comparison
#with the variational network.
def get_class_weights(y, as_dict = False):
    classes = y[:,-2].numpy()
    n_instances = np.asarray([classes.shape[0] / np.argwhere(classes==0).shape[0],
                                classes.shape[0] / np.argwhere(classes==1).shape[0],
                                classes.shape[0] / np.argwhere(classes==2).shape[0]])
    n_instances = n_instances / np.max(n_instances)
    if as_dict == False:
        return torch.from_numpy(n_instances)
    return {0:n_instances[0], 1:n_instances[1], 2:n_instances[2]}


#This function builds the specified model type on the specified encoding
#type for each split in a 5x CV on the training set only.
def score_cv(start_dir, data_type, num_epochs, model_type):
    trainx, trainy, testx, testy = load_data(start_dir, data_type)
    if trainx is None:
        raise ValueError("The data type selected by model_training_code "
                "has not yet been generated.")
    #We do not yet need testx and testy.
    del testx, testy
    #The autoencoder-generated data is flattened from a 3d tensor to a matrix.
    #All other data types are already "flat".
    if len(trainx.size()) ==3:
        trainx = trainx.flatten(1,-1)

    print("*******\n5x CV in progress for %s using %s"%(data_type, model_type))
    #Because we do not use KFold to shuffle the data no random state is needed -- 
    #this is deterministic.
    kf, indices = KFold(n_splits=5), np.arange(trainx.size()[0])
    kf.get_n_splits(indices)
    cv_mccs = []
    cv_aucs = []
    for tri, tti in kf.split(indices):
        xtrain = trainx[torch.from_numpy(indices[tri]),:]
        xval = trainx[torch.from_numpy(indices[tti]),:]
        ytrain = trainy[torch.from_numpy(indices[tri]),:]
        yval = trainy[torch.from_numpy(indices[tti]),:]
        scale_data = True
        #Very important not to try to scale onehot data...
        if data_type == "onehot":
            scale_data = False
        if model_type == "BON":
            model = BON(input_dim=trainx.size()[1])
            losses = model.trainmod(xtrain, ytrain, 
                    epochs=num_epochs, scale_data=scale_data, random_seed=0,
                    num_samples=10)
            valpreds = model.map_categorize(xval)
            valprobs = model.map_predict(xval)
        elif model_type == "FCNN":
            if scale_data:
                #The FCNN class does not auto-scale the data unlike BON so we do that here.
                #Very important obviously that we not try to scale onehot data...
                xval = (xval - torch.mean(xtrain, dim=0).unsqueeze(0)) / torch.std(xtrain, dim=0).unsqueeze(0)
                xtrain = (xtrain - torch.mean(xtrain, dim=0).unsqueeze(0)) / torch.std(xtrain, dim=0).unsqueeze(0)
            class_weights = get_class_weights(ytrain)
            model = FCNN(dropout = 0.2, input_dim = xtrain.size()[1])
            losses = model.trainmod(xtrain, ytrain, epochs=num_epochs, 
                    class_weights = class_weights, lr=0.005)
            valprobs, valpreds = model.predict(xval)
        elif model_type == "RF":
            class_weights = get_class_weights(ytrain, as_dict=True)
            model = RandomForestClassifier(n_estimators=500,
                        min_samples_split=25, min_samples_leaf=4, 
                        n_jobs=3, class_weight=class_weights,
                        random_state = 0)
            model.fit(xtrain.numpy(), ytrain.numpy()[:,-2], 
                    sample_weight=ytrain.numpy()[:,-1])
            valpreds = model.predict(xval.numpy())
            valprobs = model.predict_proba(xval.numpy())
        cv_mccs.append(mcc(yval[:,2].numpy(), valpreds))
        yval = yval.numpy()[:,2]
        yval[yval==1] = 0
        yval[yval==2] = 1
        cv_aucs.append(roc_auc_score(yval, valprobs[:,-1]))
        print("MCC: %s"%cv_mccs[-1])
        print("AUC: %s"%cv_aucs[-1])
        #Take this out if you don't want it -- I like to have a pause
        #between iterations
        time.sleep(10)
    return np.mean(cv_mccs), np.std(cv_mccs, ddof=1), np.mean(cv_aucs), np.std(cv_aucs, ddof=1)


#Convenience function for running 5x CVs.
def run_all_5x_cvs(start_dir):
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
    cv_results_dict = dict()
    #Skipping fair_esm since results were truly dismal. If you want to add it in
    #please do, but it is expensive and performs very badly.
    cv_results_dict["adapted_FCNN"] = score_cv(start_dir, "adapted", 
                    num_epochs=45, model_type = "FCNN")
    cv_results_dict["onehot_FCNN"] = score_cv(start_dir, "onehot", 
                    num_epochs=45, model_type = "FCNN")
    cv_results_dict["unirep_FCNN"] = score_cv(start_dir, "unirep", 
                    num_epochs=45, model_type = "FCNN")
    cv_results_dict["protvec_FCNN"] = score_cv(start_dir, "protvec", 
                    num_epochs=45, model_type = "FCNN")
    for data_type in ["adapted", "onehot", "protvec", "unirep"]:
        cv_results_dict[data_type + "_BON"] = score_cv(start_dir, data_type, 
                    num_epochs=45, model_type = "BON")
    cv_results_dict["adapted_RF"] = score_cv(start_dir, "adapted", 
                    num_epochs=0, model_type = "RF")
    cv_results_dict["onehot_RF"] = score_cv(start_dir, "onehot", 
                    num_epochs=0, model_type = "RF")
    os.chdir(start_dir)
    os.chdir("results_and_resources")
    with open("5x_CV_results.txt", "w+") as outf:
        outf.write("Data type_model,Average MCC on 5x CV,Std dev,"
            "Average AUC-ROC,Std dev\n")
        for key in cv_results_dict:
            outf.write(key)
            outf.write(",")
            outf.write(",".join([str(k) for k in cv_results_dict[key]]))
            outf.write("\n")

