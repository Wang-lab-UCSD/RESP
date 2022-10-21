'''This file enables straightforward reproduction of some important experiments:
training models on the training set only and evaluating them on the test
set, building the final model, scoring all sequences and extracting the
top 500, scoring the wild-type.'''

#Author: Jonathan Parkinson <jlparkinson1@gmail.com>

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy.stats import mannwhitneyu as MWU
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

from ..utilities.model_data_loader import load_model, save_model, load_data, get_antiberty_file_list
from ..model_code.traditional_nn_classification import fcnn_classifier as FCNN
from ..model_code.variational_Bayes_ordinal_reg import bayes_ordinal_nn as BON
from ..model_code.variational_Bayes_ordinal_BATCHED import batched_bayes_ordinal_nn as BATCHED_BON
from ..model_code.batched_trad_nn_classification import batched_fcnn_classifier as BATCHED_FCNN

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
    for i, letter in enumerate(full_wt_seq):
        onehot_wt[0,i,aas.index(letter)] = 1.0
    onehot_wt = torch.from_numpy(onehot_wt).float()
    x_wt = autoencoder.extract_hidden_rep(onehot_wt.cuda()).cpu()
    x_wt = x[:,29:124,:].flatten(1,-1)
    score, _ = varbayes_model.extract_hidden_rep(x_wt, use_MAP=True)
    print(f"WT MAP score: {score.numpy().flatten()[0]}")



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
        current_seq.append(f" {best_scores[i]}")
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
    xscores, _ = varbayes_model.extract_hidden_rep(x_encoded, use_MAP=True)
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
    if not as_dict:
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




def eval_train_test(start_dir, data_type, num_epochs, model_type,
        return_all_preds = False, dropout = 0.2):
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
        return_all_preds (bool): If True, return all predictions. This is
            useful for trastuzumab, where we have to calculate some additional
            properties aside from MCC.
    """
    if data_type != "antiberty":
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
    else:
        trainx_files, testx_files, trainy_files, testy_files = get_antiberty_file_list(start_dir)


    model = load_model(start_dir,
            f"{data_type}_trainset_only_{model_type}_model.ptc",
            model_type)
    print("*******\nNOW EVALUATING %s using %s"%(data_type, model_type))
    if model is None:
        #Very important not to try to scale onehot data...
        scale_data = True
        if data_type == "onehot":
            scale_data = False
        print("Model for data type %s not found...will train."%data_type)
        if model_type == "BON":
            model = BON(input_dim=trainx.size()[1], num_categories = trainy.shape[1] - 1)
            _ = model.trainmod(trainx, trainy,
                    epochs=num_epochs, scale_data=scale_data, random_seed=0,
                    num_samples=10)
            save_model(start_dir, "%s_trainset_only_%s_model.ptc"%(data_type, model_type),
                                model)
        
        elif model_type == "BATCHED_BON":
            xinit, yinit = torch.load(trainx_files[0]), torch.load(trainy_files[0])
            xinit = torch.flatten(xinit, start_dim=1)
            model = BATCHED_BON(input_dim=xinit.size()[1], num_categories = yinit.shape[1] - 1)
            _ = model.trainmod(trainx_files, trainy_files,
                    epochs=num_epochs, scale_data=scale_data, random_seed=0,
                    num_samples=10)
            save_model(start_dir, "%s_trainset_only_%s_model.ptc"%(data_type, model_type),
                                model)

        elif model_type == "BATCHED_FCNN":
            class_weights = get_batched_class_weights(trainy_files)
            xinit = torch.flatten(torch.load(trainx_files[0]), start_dim=1)
            model = BATCHED_FCNN(dropout = dropout, input_dim = xinit.size()[1])
            losses = model.trainmod(trainx_files, trainy_files, epochs=num_epochs, class_weights =
                    class_weights, lr=0.005)
            save_model(start_dir, "%s_trainset_only_%s_model.ptc"%(data_type, model_type),
                                model)

        elif model_type == "FCNN":
            class_weights = get_class_weights(trainy)
            model = FCNN(dropout = dropout, input_dim = trainx.size()[1])
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

    elif model_type == "BATCHED_BON":
        trainy, testy, trainpreds, testpreds = [], [], [], []
        for xfile, yfile in zip(testx_files, testy_files):
            x, y = torch.load(xfile), torch.load(yfile)
            testy.append(y)
            x = torch.flatten(x, start_dim=1)
            testpreds.append(model.map_categorize(x))
        for xfile, yfile in zip(trainx_files, trainy_files):
            x, y = torch.load(xfile), torch.load(yfile)
            trainy.append(y)
            x = torch.flatten(x, start_dim=1)
            trainpreds.append(model.map_categorize(x))
        trainy = torch.cat(trainy, dim=0).numpy()
        testy = torch.cat(testy, dim=0).numpy()
        trainpreds = np.concatenate(trainpreds)
        testpreds = np.concatenate(testpreds)

    elif model_type == "FCNN":
        trainpreds = model.predict(trainx)[1]
        testpreds = model.predict(testx)[1]

    elif model_type == "BATCHED_FCNN":
        trainy, testy, trainpreds, testpreds = [], [], [], []
        for xfile, yfile in zip(testx_files, testy_files):
            x, y = torch.load(xfile), torch.load(yfile)
            testy.append(y)
            x = torch.flatten(x, start_dim=1)
            testpreds.append(model.predict(x)[1])
        for xfile, yfile in zip(trainx_files, trainy_files):
            x, y = torch.load(xfile), torch.load(yfile)
            trainy.append(y)
            x = torch.flatten(x, start_dim=1)
            trainpreds.append(model.predict(x)[1])
        trainy = torch.cat(trainy, dim=0).numpy()
        testy = torch.cat(testy, dim=0).numpy()
        trainpreds = np.concatenate(trainpreds)
        testpreds = np.concatenate(testpreds)

    elif model_type == "RF":
        trainpreds = model.predict(trainx)
        testpreds = model.predict(testx)

    else:
        raise ValueError("Unrecognized model type passed to model_training_code.")
    testscore = mcc(testy[:,-2], testpreds)
    trainscore = mcc(trainy[:,-2], trainpreds)
    print("Trainscore: %s"%trainscore)
    print("Testscore: %s"%testscore)
    if not return_all_preds or data_type == "antiberty":
        return trainscore, testscore

    #The remaining code is invoked only if all predictions are desired
    #by caller, which currently is true only for the trastuzumab experiment.
    if model_type == "BON":
        testpreds = model.map_predict(testx)
    return trainscore, testscore, testpreds, testy[:,-2]






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
    alternate_encodings = False
    for expected_fname in ["adapted_testx.pt", "adapted_trainx.pt", "nonadapted_testx.pt",
            "nonadapted_trainx.pt", "onehot_testx.pt", "onehot_trainx.pt", "protvec_testx.pt",
            "protvec_trainx.pt", "testy.pt", "trainy.pt", "unirep_testx.pt", "unirep_testy.pt",
            "unirep_trainx.pt", "unirep_trainy.pt", "fair_esm_trainx.pt", "fair_esm_testx.pt"]:
        if expected_fname not in fnames:
            print("The data has not been encoded using all of the expected encoding types. This "
                    "step is required before train test evaluation.")
            return
    if "ablang_trainx.pt" in fnames and "antiberty_embeds" in fnames:
        alternate_encodings = True

    if action_to_take == "traintest_eval":
        test_results_dict = dict()
        for data_type in ["s10", "s25", "s50", "adapted", "nonadapted", "onehot",
                "protvec", "unirep", "fair_esm"]:
            test_results_dict[data_type + "_BON"] = eval_train_test(start_dir, data_type, 
                    num_epochs=40, model_type = "BON")

        if alternate_encodings:
            #Ablang requires more iterations for convergence (although the improvement
            #in performance going from 40 epochs to 60 is very slight, almost negligible)
            test_results_dict["ablang_BON"] = eval_train_test(start_dir, "ablang",
                    num_epochs=60, model_type = "BON")
            test_results_dict["ablang_FCNN"] = eval_train_test(start_dir, "ablang", 
                    num_epochs=60, model_type = "FCNN")
            #Antiberty performed horribly, but modifying some of the hyperparameters (more
            #epochs, less dropout) didn't seem to help.
            test_results_dict["antiberty_BON"] = eval_train_test(start_dir, "antiberty",
                    num_epochs=40, model_type = "BATCHED_BON")
            test_results_dict["antiberty_FCNN"] = eval_train_test(start_dir, "antiberty", 
                    num_epochs=40, model_type = "BATCHED_FCNN", dropout = 0.0)


        for data_type in ["s10", "s25", "s50", "adapted", "onehot", "unirep", "fair_esm", "protvec"]:
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
    #data using a variety of different schemes and should make sure all are
    #present before proceeding.
    for expected_fname in ["adapted_testx.pt", "adapted_trainx.pt"]:
        if expected_fname not in fnames:
            print("The data has not been encoded using all of the expected encoding types. This "
                    "step is required before train test evaluation.")
            return
    #Initial experiments on the validation set suggested -- interestingly --
    #that more epochs are required to reach full performance on trastuzumab
    #than on PDL1. We doubled the number of epochs to be sure we were reaching
    #convergence here.
    test_mcc, train_mcc, testpreds, testy = eval_train_test(start_dir, "adapted",
                    num_epochs=80, model_type = "BON", return_all_preds = True)

    roc = roc_auc_score(testy, testpreds)

    os.chdir(project_dir)
    os.chdir("results_and_resources")
    with open("trastuzumab.txt", "w+") as outf:
        outf.write("Data type_model,MCC train,MCC test,ROC_AUC_test\n")
        outf.write(f"NA,{test_mcc},{train_mcc},{roc}\n")


def score_trastuzumab(project_dir):
    """Scores the 55 sequences selected for experimental evaluation in
    the original paper.

    Args:
        project_dir (str): A filepath to the project directory.
    """
    os.chdir(os.path.join(project_dir, "trastuzumab_data"))
    start_dir = os.getcwd()

    model = load_model(start_dir,
            "adapted_trainset_only_BON_model.ptc", "BON")
    if model is None:
        raise ValueError("The trastuzumab model has not yet been trained.")
    os.chdir(start_dir)
    os.chdir("encoded_data")
    if "experimental_seqs.pt" not in os.listdir() or "experimental_kd.pt" \
            not in os.listdir():
        raise ValueError("The experimental seqs have not been encoded yet.")

    kd_values = torch.load("experimental_kd.pt").numpy()
    train_x = torch.load("adapted_trainx.pt").flatten(1,-1)
    test_x = torch.load("adapted_testx.pt").flatten(1,-1)
    x_data = torch.load("experimental_seqs.pt").flatten(1, -1)
    train_y = torch.load("trainy.pt")
    test_y = torch.load("testy.pt").numpy()

    train_scores = model.extract_hidden_rep(train_x, use_MAP=True)[0].numpy().flatten()
    score = model.extract_hidden_rep(x_data, use_MAP=True)[0].numpy().flatten()

    os.chdir(start_dir)
    os.chdir("results_and_resources")
    plt.style.use("bmh")
    labels = ["Nonbinder, training set" if s == 0 else "Binder, training set" for
            s in train_y[:,1].tolist()]
    sns.displot(x=train_scores, hue=labels)
    plt.xlabel("Model assigned score")
    plt.title("Model assigned score for training set sequences\n"
            "from Mason et al.")
    plt.savefig("model_training_scores.png", bbox_inches = "tight")
    plt.close()

    _ = plt.hist(score, label="Experimentally evaluated seqs")
    matching_x, matching_y = np.full((50),score[-1]), np.linspace(0, 10, 50)
    plt.plot(matching_x, matching_y, label="Trastuzumab_score")
    plt.xlabel("Model assigned score")
    plt.title("Model assigned score for experimentally evaluated\nsequences "
            "from Mason et al.")
    plt.legend()
    plt.savefig("model_exp_scores.png", bbox_inches = "tight")
    plt.close()
    #Plot the score for trastuzumab


    testpreds = model.map_categorize(test_x)
    mismatches = ["Correct prediction" if p == gt else "Incorrect prediction" for
            p, gt in zip(testpreds.tolist(), test_y[:,1].tolist())]
    score, stdev = model.extract_hidden_rep(test_x, num_samples=250,
                        random_seed = 123)
    score, stdev = score.numpy(), stdev.numpy()
    sns.boxplot(x=mismatches, y=np.log(stdev), notch=True)
    plt.title("Model uncertainty for correct and incorrect test set predictions.")
    plt.ylabel("Log standard deviation")
    plt.savefig("Uncertainty_vs_pred.png", bbox_inches = "tight")
    plt.close()

    correct_stdev = [s for s, m in zip(stdev.tolist(), mismatches) if m == "Correct prediction"]
    incorrect_stdev = [s for s, m in zip(stdev.tolist(), mismatches) if m == "Incorrect prediction"]
    print(MWU(correct_stdev, incorrect_stdev))
