'''These functions are used to load and save models and load training or test
data generated using a specific encoding. These functions are used by the
model_training_code and simulated_annealing modules. It also includes code
for generating dictionaries used for mapping input sequences to Chothia
numbering -- crucial since the autoencoders were trained on Chothia-numbered
input.'''

#Author: Jonathan Parkinson <jlparkinson1@gmail.com>

import torch, os, sys, pickle
from ..model_code.traditional_nn_classification import fcnn_classifier as FCNN
from ..model_code.variational_Bayes_ordinal_reg import bayes_ordinal_nn as BON
from ..model_code.task_adapted_autoencoder import TaskAdaptedAutoencoder as TAE
from ..model_code.unadapted_autoencoder import UnadaptedAutoencoder as UAE
from ..model_code import task_adapted_ae_linlayer_subsampled
from ..model_code.task_adapted_ae_linlayer_subsampled import TaskAdaptedAeLinlayerSampled

#A list of numbered positions for Chothia antibody sequence numbering.
chothia_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11',
            '12', '13', '14', '15', '16', '17', '18', '19', '20', '21',
            '22', '23', '24', '25', '26', '27', '28', '29', '30', '31',
            '31A', '31B', '32', '33', '34', '35', '36', '37', '38', '39',
            '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50',
            '51', '52', '52A', '52B', '52C', '53', '54', '55', '56', '57',
            '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68',
            '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79',
            '80', '81', '82', '82A', '82B', '82C', '83', '84', '85', '86',
            '87', '88', '89', '90', '91', '92', '93', '94', '95', '96',
            '97', '98', '99', '100', '100A', '100B', '100C', '100D', '100E',
            '100F', '100G', '100H', '100I', '100J', '100K', '101', '102',
            '103', '104', '105', '106', '107', '108', '109', '110', '111',
            '112', '113']

def gen_anarci_dict(start_dir):
    """This function generates dictionaries used to map input sequences to Chothia
    numbering, so that blanks can be inserted in appropriate places.

    Args:
        start_dir (str): A filepath to the starting directory.

    Returns:
        position_dict (dict): A dictionary that maps each position in a
            PDL1 sequence to the corresponding Chothia numbered position.
        unused_positions (list): A list of the Chothia numbered positions
            that are not used for PDL1.
    """
    os.chdir(start_dir)
    os.chdir("results_and_resources")
    position_dict, positions_used = dict(), set()
    positions_used = set()
    with open("wt_anarci_numbering.txt") as input_file:
        lines = [line for line in input_file if line.startswith("H ")]
    for i, line in enumerate(lines):
        if len(line.strip().split()) == 4:
            position = line.split()[1] + line.split()[2]
        else:
            position = line.split()[1]
        position_dict[i] = chothia_list.index(position)
        positions_used.add(position)
    unused_positions = [i for i in range(len(chothia_list)) if 
                        chothia_list[i] not in positions_used]
    os.chdir(start_dir)
    return position_dict, unused_positions



def load_model(start_dir, model_filename, model_type):
    """This function loads the appropriate model for the specified data type
    if it is present in the results_and_resources/trained_models dir;
    if not, it returns None, indicating the corresponding model needs
    to be trained.

    Args:
        start_dir (str): A path to the starting directory.
        model_filename (str): The name of the saved model to look for.
        model_type (str): The type of model. Determines how the model
            is loaded and set up.

    Returns:
        model: The model object.
    """
    os.chdir(start_dir)
    os.chdir(os.path.join("results_and_resources", "trained_models"))
    model = None
    #Random forest models are the only type saved with pickle not PyTorch, so
    #they require some special handling.
    if model_type == "RF":
        model_filename = model_filename.split(".ptc")[0] + ".pk"
    if model_filename in os.listdir():
        if model_type == "RF":
            with open(model_filename, "rb") as inf:
                model = pickle.load(inf)
            os.chdir(start_dir)
            return model

        model_state_dict = torch.load(model_filename)
        if model_type == "BON":
            input_dim = model_state_dict["n1.weight_means"].size()[0]
            num_categories = model_state_dict["output_layer.fixed_thresh"].shape[0] + 1
            #This is a hack for backwards compatibility. TODO: Update this
            if num_categories == 2:
                model = BON(input_dim = input_dim, num_categories = num_categories)
            else:
                model = BON(input_dim = input_dim)
            model.load_state_dict(model_state_dict)
        elif model_type == "adapted":
            model = TAE()
            model.load_state_dict(model_state_dict)
        elif model_type == "nonadapted":
            model = UAE()
            model.load_state_dict(model_state_dict)
        elif model_type == "FCNN":
            input_dim = model_state_dict["n1.weight"].size()[1]
            model = FCNN(input_dim=input_dim)
            model.load_state_dict(model_state_dict)
        elif model_type == "subsample":
            model = TaskAdaptedAeLinlayerSampled()
            model.load_state_dict(model_state_dict)
        else:
            raise ValueError("Unrecognized model type supplied to Utilities.")
    os.chdir(start_dir)
    return model


def save_model(start_dir, model_filename, model):
    """This function saves the model to the results_and_resources/trained_models
    location. If the model is already present an error message is printed but no
    exception is raised since this is non-fatal.

    Args:
        start_dir (str): A path to the start directory.
        model_filename (str): A name to save the model under.
        model: The model object.
    """
    os.chdir(start_dir)
    os.chdir(os.path.join("results_and_resources", "trained_models"))
    if model_filename in os.listdir():
        print("Utilities was asked to overwrite an existing trained model. This is not recommended. "
                "If you want to do this you will have to do so manually (by removing the model you would like to "
                "overwrite from the results_and_resources/trained_models directory). For now the existing "
                "model will be used.")
        return
    else:
        #Random forest models are not PyTorch-based and thus are saved using pickle instead.
        if "_RF_" in model_filename:
            with open(model_filename, "wb") as outf:
                pickle.dump(model, outf)
        else:
            torch.save(model.state_dict(), model_filename)
    os.chdir(start_dir)


def load_data(start_dir, data_type):
    """This function loads the training and test x and y data for a specified data type.
    If that data has not been generated yet, it returns None for all.
    Data files are returned as the list [trainset_x, trainset_y,
            testset_x, testset_y].

    Args:
        start_dir (str): The path to the starting directory.
        data_type (str): The type of data to load ("unirep" or some other).

    Returns:
        data_files (list): A list of train_x, train_y, test_x and test_y
            data, all as PyTorch tensors.
    """
    os.chdir(start_dir)
    os.chdir("encoded_data")
    data_files = []
    #Unirep data is handled differently since it could not be generated from the one-hot data but
    #rather had to be encoded from the raw sequences, so the y-values are generated separately.
    if data_type == "unirep":
        raw_files = ["unirep_trainx.pt", "unirep_trainy.pt", "unirep_testx.pt",
                "unirep_testy.pt"]
    else:
        raw_files = ["%s_trainx.pt"%data_type, "trainy.pt", "%s_testx.pt"%data_type,
                    "testy.pt"]
    for raw_file in raw_files:
        if raw_file not in os.listdir():
            os.chdir(start_dir)
            return [None, None, None, None]
        data_files.append(torch.load(raw_file))

    os.chdir(start_dir)
    return data_files
