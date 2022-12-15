"""Processes the raw seqs from the trastuzumab Nature Biomed Engineering
paper so that they can be encoded for uptake by the autoencoder or for
use by other models."""
import os
from copy import deepcopy
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from ..model_code.task_adapted_autoencoder import TaskAdaptedAutoencoder as TAE
from ..utilities.model_data_loader import gen_anarci_dict, load_data, load_model
#TODO: Move aas and wildtype to a constants file.
aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
       'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']

#A list of numbered positions for Chothia antibody sequence numbering.
#TODO: Transfer to constants file.
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


def data_split_adj(Ag_pos, Ag_neg, fraction):
    """
    NOTE: This function is copied intact from 
    https://github.com/dahjan/DMS_opt to ensure we use
    the same train-test split methodology.

    Create a collection of the data set and split into the
    training set and two test sets. Data set is adjusted to
    match the specified class split fraction, which determines
    the fraction of Ag+ sequences.

    Parameters
    ---
    Ag_pos: Dataframe of the Ag+ data set
    Ag_neg: Dataframe of the Ag- data set
    fraction: The desired fraction of Ag+ in the data set
    """

    class Collection:
        def __init__(self, **kwds):
            self.__dict__.update(kwds)

    # Calculate data sizes based on ratio
    data_size_pos = len(Ag_pos)/fraction
    data_size_neg = len(Ag_neg)/(1-fraction)

    # Adjust the length of the data frames to meet the ratio requirement
    if len(Ag_pos) <= len(Ag_neg):
        if data_size_neg < data_size_pos:
            Ag_pos1 = Ag_pos[0:int((data_size_neg*fraction))]
            Ag_neg1 = Ag_neg
            Unused = Ag_pos[int((data_size_neg*fraction)):len(Ag_pos)]

        if data_size_neg >= data_size_pos:
            Ag_pos1 = Ag_pos[0:int((data_size_pos*(fraction)))]
            Ag_neg1 = Ag_neg[0:int((data_size_pos*(1-fraction)))]
            Unused = pd.concat(
                [Ag_pos[int((data_size_pos*fraction)):len(Ag_pos)],
                 Ag_neg[int((data_size_pos*(1-fraction))):len(Ag_neg)]]
            )
    else:
        if data_size_pos < data_size_neg:
            Ag_pos1 = Ag_pos
            Ag_neg1 = Ag_neg[0:(int(data_size_pos*(1-fraction)))]
            Unused = Ag_pos[int((data_size_pos*fraction)):len(Ag_pos)]

        if data_size_pos >= data_size_neg:
            Ag_pos1 = Ag_pos[0:int((data_size_neg*(fraction)))]
            Ag_neg1 = Ag_neg[0:int((data_size_neg*(1-fraction)))]
            Unused = pd.concat(
                [Ag_pos[int((data_size_neg*fraction)):len(Ag_pos)],
                 Ag_neg[int((data_size_neg*(1-fraction))):len(Ag_neg)]]
            )
    # Combine the positive and negative data frames
    # Original function did not supply a random seed here, fixed
    Ag_combined = pd.concat([Ag_pos1, Ag_neg1])
    #Ag_combined = Ag_combined.drop_duplicates(subset='AASeq')
    Ag_combined = Ag_combined.sample(frac=1, random_state=0).reset_index(drop=True)

    # 70%/30% training test data split
    # Original function did not supply a random seed here, fixed
    idx = np.arange(0, Ag_combined.shape[0])
    idx_train, idx_test = train_test_split(
        idx, stratify=Ag_combined['AgClass'], test_size=0.3,
        random_state=0
    )
    
    # 50%/50% test validation data split
    # Original function did not supply a random seed here, fixed
    idx2 = np.arange(0, idx_test.shape[0])
    idx_val, idx_test2 = train_test_split(
        idx2, stratify=Ag_combined.iloc[idx_test, :]['AgClass'], test_size=0.5,
        random_state=0
    )

    # Create collection
    Seq_Ag_data = Collection(
        train=Ag_combined.iloc[idx_train, :],
        val=Ag_combined.iloc[idx_test, :].iloc[idx_val, :],
        test=Ag_combined.iloc[idx_test, :].iloc[idx_test2, :],
        complete=Ag_combined
    )
    #Add unused to training set and shuffle
    Seq_Ag_data.train = pd.concat([Seq_Ag_data.train, Unused])
    Seq_Ag_data.train = Seq_Ag_data.train.sample(frac=1, random_state=0)
    return Seq_Ag_data, Unused



def build_trastuzumab_chothia_dict():
    """Maps the trastuzumab aas to the Chothia numbering system and inserts
    blanks where appropriate, to prepare the data for the autoencoder
    and/or for other encoding schemes.

    Returns:
        wt (list): A list of the amino acids in the WT, with blanks inserted
            where appropriate.
        cutout_positions (tuple): A tuple with (first CDRH3 position, last
            CDR H3 position).
    """
    chothia_encoding = pd.read_csv("trastuzumab_anarci.txt_H.csv")
    col_list = chothia_encoding.columns.tolist()
    start_pos = col_list.index("1")
    position_dict, wt = {}, []
    for i, col in enumerate(col_list[start_pos:]):
        position_dict[col] = chothia_encoding[col][0]
    for key in chothia_list:
        if key in position_dict:
            wt.append(position_dict[key])
        else:
            wt.append("-")
    cutout_positions = (chothia_list.index("95"), chothia_list.index("102"))
    return wt, cutout_positions


def sequence_weighting(mHER_pos, mHER_neg, wt, cutout_positions):
    """Some of the sequences listed are listed in both the
    positive binders category and negative binders category.
    This is a common problem and is one we resolve in our
    study through weighting. However in the Nature Biotech
    study, these sequences seem to have been placed in both
    categories so that the model is asked to classify them
    as...both. To avoid inconsistency, we adopt the same
    approach here, but we use our weighting scheme, so
    that the model downweights inconsistent sequences such
    as these. This function calculates the weights and also
    inserts the mutated region into the full trastuzumab sequence.

    Args:
        mHER_pos (pd.DataFrame): A pandas data frame with
            binders.
        mHER_neg (pd.DataFrame): A pandas data frame with
            nonbinders.

    Returns:
        mHER_pos (pd.DataFrame): The input mHER_pos, augmented
            with weights etc.
        mHER_neg (pd.DataFrame); The input mHER_neg, augmented
            with weights etc.
    """
    sequence_dict = {}
    for seq, freq in zip(mHER_neg.AASeq.tolist(),
            mHER_neg.Count.tolist()):
        sequence_dict[seq] = np.asarray([float(freq), 0, 0, 0])
    for seq, freq in zip(mHER_pos.AASeq.tolist(),
            mHER_pos.Count.tolist()):
        if seq in sequence_dict:
            sequence_dict[seq][1] = float(freq)
        else:
            sequence_dict[seq] = np.asarray([0, float(freq), 0, 0])

    for seq, freq_data in sequence_dict.items():
        weight = (freq_data.max() + 1.0) / (freq_data.sum() + 3.0)
        sequence_dict[seq][2] = weight
        sequence_dict[seq][3] = np.argmax(sequence_dict[seq][0:2])

    return prep_seq_dataframe(sequence_dict, wt, cutout_positions)


def prep_seq_dataframe(seq_dict, wt, cutoff_pos):
    """Given a sequence dict mapping mutant
    sequences to frequencies & weights, the full wild type sequence,
    and the region into which the mutatant sequence should be inserted,
    this function appends two new columns to the dataframe, one
    containing sequence weight and the other containing the full
    sequence with the mutated CDRH3 patched in.

    Args:
        seq_dict (dict): A dict mapping sequences to numpy arrays
            containing (frequency in nonbinder category, frequency
            in binder category, sequence weight)
        input_df (pd.DataFrame): A pandas data frame that will be modified
            and returned by adding the two appropriate columns.
        wt (list): The full WT sequence, with chothia numbered blanks
            inserted.
        cutoff_pos (tuple): The region containing CDR H3.

    Returns:
        output_df (pd.DataFrame): The input data frame with the two columns
            described above appended.
    """
    pos_df = {"weight":[], "AgClass":[], "Full_Sequence":[]}
    neg_df = {"weight":[], "AgClass":[], "Full_Sequence":[]}
    for seq, values in seq_dict.items():
        if values[2] < 0.51:
            continue
        expanded_seq = deepcopy(wt)
        expanded_seq[cutoff_pos[0]:cutoff_pos[1]] = list(seq)
        category = values[3]
        if category == 1:
            pos_df["weight"].append(values[2])
            pos_df["AgClass"].append(values[3])
            pos_df["Full_Sequence"].append("".join(expanded_seq))
        elif category == 0:
            neg_df["weight"].append(values[2])
            neg_df["AgClass"].append(values[3])
            neg_df["Full_Sequence"].append("".join(expanded_seq))
        else:
            raise ValueError("Unexpected category encountered.")
    pos_df = pd.DataFrame.from_dict(pos_df)
    neg_df = pd.DataFrame.from_dict(neg_df)
    pos_df.sort_values(by="weight", ascending=False, inplace=True)
    neg_df.sort_values(by="weight", ascending=False, inplace=True)
    return pos_df, neg_df


def onehot_encode_dataset(input_df, wt):
    """One-hot encodes the data, either to save directly or to encode
    using the autoencoder or some other representation."""
    onehot_encoded = torch.zeros((input_df.shape[0], len(wt), 21))
    yvalues = np.zeros((input_df.shape[0], 3))
    categories = input_df.AgClass.tolist()
    weights = input_df.weight.tolist()
    for i, seq in enumerate(input_df.Full_Sequence.tolist()):
        seq_list = list(seq)
        for j, letter in enumerate(seq_list):
            onehot_encoded[i,j,aas.index(letter)] = 1.0
        yvalues[i,0] = categories[i]
        yvalues[i,1] = categories[i]
        yvalues[i,2] = weights[i]

    yvalues = torch.from_numpy(yvalues)
    return onehot_encoded, yvalues



def prep_exp_valid_seqs(project_dir, wt, cutout_positions):
    """Encodes the experimentally validated seqs from the Nature Biotech
    paper for later scoring."""
    os.chdir(project_dir)
    adapt_model = load_model(project_dir, "TaskAdapted_Autoencoder.ptc", "adapted")
    os.chdir("trastuzumab_data")
    start_dir = os.getcwd()

    os.chdir(start_dir)
    exp_seqs = pd.read_csv("selected_55.txt")
    trimmed_seqs = [s[3:-2] for s in exp_seqs["Seq"].tolist()]
    trimmed_seqs = [wt[:cutout_positions[0]] + list(s) + wt[cutout_positions[1]:]
            for s in trimmed_seqs]
    trimmed_seqs = ["".join(s) for s in trimmed_seqs]
    exp_seqs["Full_Sequence"] = trimmed_seqs
    #Enter dummy information so we can use the one hot encoding function unmodified.
    #We will save the Kd in place of the weights.
    exp_seqs["AgClass"] = np.zeros((exp_seqs.shape[0]))
    exp_seqs["weight"] = exp_seqs["Kd"].values

    x_exp, y_exp = onehot_encode_dataset(exp_seqs, wt)
    y_exp = y_exp[:,2]
    encoded_x_exp = adapt_model.extract_hidden_rep(x_exp)
    encoded_x_exp = encoded_x_exp[:,cutout_positions[0]-12:,:]
    os.chdir("encoded_data")
    torch.save(encoded_x_exp, "experimental_seqs.pt")
    torch.save(y_exp, "experimental_kd.pt")

    os.chdir(start_dir)



def encode_trastuzumab_seqs(start_dir):
    """Encodes the raw mutant sequences in an appropriate form for input
    into the autoencoder or other encoding scheme.

    Args:
        start_dir (str): A filepath to the project directory.
    """
    os.chdir(start_dir)
    adapt_model = load_model(start_dir, "TaskAdapted_Autoencoder.ptc", "adapted")
    os.chdir(os.path.join("trastuzumab_data", "raw_data"))

    wt, cutout_positions = build_trastuzumab_chothia_dict()
    mHER_neg = pd.read_csv("mHER_H3_AgNeg.csv")
    mHER_pos = pd.read_csv("mHER_H3_AgPos.csv")
    mHER_pos, mHER_neg = sequence_weighting(mHER_pos, mHER_neg, wt,
                    cutout_positions)
    
    #Retrieve training, validation and test datasets using the
    #same splitting methodology as in the original Nature Biotech
    #paper.
    mHER_all_adj, unused_seq = data_split_adj(
        mHER_pos, mHER_neg, fraction=0.5)

    os.chdir(os.path.join(start_dir, "trastuzumab_data"))
    if "encoded_data" not in os.listdir():
        os.mkdir("encoded_data")
    if "results_and_resources" not in os.listdir():
        os.mkdir("results_and_resources")
        os.chdir("results_and_resources")
        os.mkdir("trained_models")
        os.chdir("..")
    os.chdir("encoded_data")
    onehot_train, ytrain = onehot_encode_dataset(mHER_all_adj.train, wt)
    onehot_valid, yvalid = onehot_encode_dataset(mHER_all_adj.val, wt)
    onehot_test, ytest = onehot_encode_dataset(mHER_all_adj.test, wt)

    encoded_trainx = adapt_model.extract_hidden_rep(onehot_train)
    encoded_testx = adapt_model.extract_hidden_rep(onehot_test)
    encoded_validx = adapt_model.extract_hidden_rep(onehot_valid)
    encoded_trainx = encoded_trainx[:,cutout_positions[0]-12:,:]
    encoded_validx = encoded_validx[:,cutout_positions[0]-12:,:]
    encoded_testx = encoded_testx[:,cutout_positions[0]-12:,:]

    torch.save(encoded_trainx, "adapted_trainx.pt")
    torch.save(encoded_testx, "adapted_testx.pt")
    torch.save(encoded_validx, "adapted_validx.pt")
    torch.save(ytrain, "trainy.pt")
    torch.save(ytest, "testy.pt")
    torch.save(yvalid, "validy.pt")

    prep_exp_valid_seqs(start_dir, wt, cutout_positions)
