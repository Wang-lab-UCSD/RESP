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

import time
import os
import sys
import numpy as np
import torch
from ..utilities.model_data_loader import load_data, get_antiberty_file_list
from torch.utils.data import random_split
from sklearn.model_selection import KFold
from sklearn.metrics import matthews_corrcoef as mcc, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from ..model_code.traditional_nn_classification import fcnn_classifier as FCNN
from ..model_code.variational_Bayes_ordinal_reg import bayes_ordinal_nn as BON
from ..model_code.variational_Bayes_ordinal_BATCHED import batched_bayes_ordinal_nn as BATCHED_BON
from ..model_code.batched_trad_nn_classification import batched_fcnn_classifier as BATCHED_FCNN
from sklearn.model_selection import KFold

#TODO: Move full_wt_seq and aas to a constants file.
full_wt_seq = ('EVQLVESGGGLVQPGGSLRLSCAASGFTFSD--SWIHWVRQAPGKGLEWVAWISP--'
                'YGGSTYYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCARRHWPGGF----------DYWGQGTLVTVSS')

aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
       'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']


def get_class_weights(y, as_dict = False):
    """This function is used to generate class weights for the random forest and
    fully connected classifiers, which are generated as baselines for comparison
    with the variational network.

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


def get_batched_class_weights(yfiles):
    """This function is used to generate class weights for the batched fully
    connected classifier.

    Args:
        yfiles (list): A list of the y data files.

    Returns:
        class_weights: A tensor.
    """
    classes = [torch.load(yfile) for yfile in yfiles]
    classes = torch.cat(classes, dim=0)
    return get_class_weights(classes)


def score_cv(start_dir, data_type, num_epochs, model_type, dropout = 0.2):
    """This function builds the specified model type on the specified encoding
    type for each split in a 5x CV on the training set only.

    Args:
        start_dir (str): A path to the project directory.
        data_type (str): The type of data (e.g. one hot encoded etc).
        num_epochs (int): The number of training epochs.
        model_type (str): The type of model (e.g. fully connected etc).

    Returns:
        mean_MCC (float): The matthews corrcoef.
        MCC_std (float): The standard deviation on matthews corrcoef.
        mean_auc (float): The AUC-ROC.
        auc_std (float): The standard deviation on AUC-ROC.
    """
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
            model = FCNN(dropout = dropout, input_dim = xtrain.size()[1])
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



def batched_score_cv(start_dir, data_type, num_epochs, model_type):
    """This function is similar to score_cv, but is intended only for datasets
    too large to load to memory (e.g. antiberty).

    Args:
        start_dir (str): A path to the project directory.
        data_type (str): The type of data (e.g. one hot encoded etc).
        num_epochs (int): The number of training epochs.
        model_type (str): The type of model (e.g. fully connected etc).

    Returns:
        mean_MCC (float): The matthews corrcoef.
        MCC_std (float): The standard deviation on matthews corrcoef.
        mean_auc (float): The AUC-ROC.
        auc_std (float): The standard deviation on AUC-ROC.
    """
    xfiles, _, yfiles, _ = get_antiberty_file_list(start_dir)

    print("*******\n5x CV in progress for %s using %s"%(data_type, model_type))
    rng = np.random.default_rng(123)
    idx = rng.permutation(len(xfiles))
    indices = np.split(idx, 5)
    cv_mccs, cv_aucs = [], []

    for i, tti in enumerate(indices):
        valx, valy = [xfiles[i] for i in tti.tolist()], \
                [yfiles[i] for i in tti.tolist()]
        tri = [indices[j] for j in range(len(indices)) if j != i]
        tri = np.concatenate(tri)
        trainx, trainy = [xfiles[i] for i in tri.tolist()], \
                [yfiles[i] for i in tri.tolist()]
        scale_data = True
        #Very important not to try to scale onehot data...
        if data_type == "onehot":
            scale_data = False

        if model_type == "BON":
            xinit, yinit = torch.load(trainx[0]), torch.load(trainy[0])
            xinit = torch.flatten(xinit, start_dim=1)
            model = BATCHED_BON(input_dim = xinit.size()[1],
                    num_categories = yinit.shape[1] - 1)
            _ = model.trainmod(trainx, trainy,
                    epochs=num_epochs, scale_data=scale_data,
                    random_seed=0, num_samples=10)
            yval, valpreds, valprobs = [], [], []
            for xfile, yfile in zip(valx, valy):
                x, y = torch.load(xfile), torch.load(yfile)
                yval.append(y)
                x = torch.flatten(x, start_dim=1)
                valpreds.append(model.map_categorize(x))
                valprobs.append(model.predict(x))
        elif model_type == "FCNN":
            class_weights = get_batched_class_weights(trainy)
            xinit, yinit = torch.load(trainx[0]), torch.load(trainy[0])
            xinit = torch.flatten(torch.load(trainx[0]), start_dim=1)
            model = BATCHED_FCNN(dropout = dropout, input_dim = xinit.size()[1])
            losses = model.trainmod(trainx, trainy,
                    epochs=num_epochs, class_weights = class_weights,
                    lr=0.005)
            yval, valpreds, valprobs = [], [], []
            for xfile, yfile in zip(valx, valy):
                x, y = torch.load(xfile), torch.load(yfile)
                yval.append(y)
                x = torch.flatten(x, start_dim=1)
                probs, preds = model.predict(x)
                valpreds.append(preds)
                valprobs.append(probs)

        yval = torch.cat(yval, dim=0)
        valpreds, valprobs = np.concatenate(valpreds), \
                np.concatenate(valprobs, axis=0)
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




def run_all_5x_cvs(start_dir):
    """Convenience function for running 5x CVs for all data types
        and models at the same time."""
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
    cv_results_dict["ablang_FCNN"] = score_cv(start_dir, "ablang",
                    num_epochs=60, model_type = "FCNN")
    cv_results_dict["antiberty_FCNN"] = score_cv(start_dir, "antiberty",
                    num_epochs=60, model_type = "FCNN", dropout = 0.25)
    cv_results_dict["ablang" + "_BON"] = score_cv(start_dir, "ablang",
                    num_epochs=60, model_type = "BON")
    cv_results_dict["antiberty" + "_BON"] = score_cv(start_dir, "antiberty",
                    num_epochs=60, model_type = "BON")
    for data_type in ["adapted", "onehot", "protvec", "unirep"]:
        cv_results_dict[data_type + "_BON"] = score_cv(start_dir, data_type,
                    num_epochs=45, model_type = "BON")
    cv_results_dict["adapted_RF"] = score_cv(start_dir, "adapted",
                    num_epochs=0, model_type = "RF")
    cv_results_dict["onehot_RF"] = score_cv(start_dir, "onehot",
                    num_epochs=0, model_type = "RF")
    os.chdir(start_dir)
    os.chdir("results_and_resources")
    if "5x_CV_results.txt" not in os.listdir():
        with open("5x_CV_results.txt", "w+") as outf:
            outf.write("Data type_model,Average MCC on 5x CV,Std dev,"
                "Average AUC-ROC,Std dev\n")

    with open("5x_CV_results.txt", "a+") as outf:
        for key in cv_results_dict:
            outf.write(key)
            outf.write(",")
            outf.write(",".join([str(k) for k in cv_results_dict[key]]))
            outf.write("\n")
