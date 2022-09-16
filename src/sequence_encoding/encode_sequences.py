'''Encodes the RH01, RH02 and RH03 amino acid sequences as PyTorch tensors.
As a historical artefact of the way the pipeline was developed (because the one-
hot encoding is the input to the autoencoders), the AA seqs
are encoded first as one-hot tensors,  which are train-test split, then these
are converted to other representations. This requires some awkward contortions
in a few places, but this logic has been retained so that you can reproduce
the experiments we performed in the way we originally performed them. Future
pipelines should train-test split the aa sequences first.

We generate the following encodings:
One-hot, task-adapted autoencoder, unadapted autoencoder, protvec, Unirep.
All of these are saved as tensors -- for this moderate-sized dataset, this is
a convenient approach.

Note that before encoding we use Chothia numbering to insert blanks at appropriate
locations. This is necessary since the autoencoders were trained on Chothia-numbered
antibody sequences.'''

#Author: Jonathan Parkinson <jlparkinson1@gmail.com>

import os, numpy as np, Bio, torch, sys
from Bio.Seq import Seq
from ..model_code.task_adapted_autoencoder import TaskAdaptedAutoencoder as TAE
from ..model_code.unadapted_autoencoder import UnadaptedAutoencoder as UAE
from ..utilities.model_data_loader import gen_anarci_dict, load_data, load_model


#TODO: Move aas and wildtype to a constants file.
aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
       'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']
wildtype = ('EVQLVESGGGLVQPGGSLRLSCAASGFTFSDSWIHWVRQAPGKGLE'
            'WVAWISPYGGSTYYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCARRHWPGGFDYWGQGTLVTVSS')

def one_hot_encode(start_dir, position_dict, unused_positions):
    """This key function converts RH01, RH02, RH03 to one-hot encoded Chothia numbered
    sequences and train-test splits them. The y-tensors are shared among all data types
    and contain 4 columns. The first two are one-hot encoded to indicate whether
    sequence is > RH01 and whether sequence is >RH02. The third is a numerical category
    indicator (0=RH01, 1=RH02, 2=RH03) and the fourth is the sequence weight, based on
    the relative frequency in assigned category and other categories (see the paper).

    Args:
        start_dir (str): A filepath to the project directory.
        position_dict (dict): A dictionary mapping positions in the sequence to
            Chothia numbering positions. Can be obtained from Utilities.
        unusued_positions (list): A list of chothia positions that are not
            used. Can be obtained from Utilities.

    Returns:
        trainx (tensor): A PyTorch tensor of train x data.
        trainy (tensor): A PyTorch tensor of train y data.
        testx (tensor): A PyTorch tensor of test x data.
        testy (tensor): A PyTorch tensor of test y data.
    """
    seqdict, total_seqs = dict(), 0
    unclear_category, unexpected_mutation = 0, 0
    os.chdir("encoded_data")
    filelist = ["rh01_sequences.txt", "rh02_sequences.txt",
                    "rh03_sequences.txt"]
    for k, filename in enumerate(filelist):
        file_handle = open(filename)
        for curr_line in file_handle:
            full_seq_aa = curr_line.split('\t',1)[0]
            frequency = int(curr_line.split('\t', 1)[1].strip())
            if full_seq_aa == wildtype:
                continue
            if full_seq_aa[0:29] != wildtype[0:29] or full_seq_aa[-8:] != wildtype[-8:]:
                unexpected_mutation += 1
                continue
            if full_seq_aa not in seqdict:
                seqdict[full_seq_aa] = np.zeros((len(filelist)))
            seqdict[full_seq_aa][k] = frequency
        file_handle.close()
    print('Total seqs found: %s'%(len(seqdict)))
    print("%s seqs with unexpected mutations"%unexpected_mutation)

    os.chdir(start_dir)
    x, y = [], []
    for key in seqdict:
        x_array = np.zeros((1,132,21))
        y_array = np.zeros((len(filelist)+1))
        for j, letter in enumerate(key):
            x_array[0,position_dict[j],aas.index(letter)] = 1.0
        for unused_position in unused_positions:
            x_array[0, unused_position, 20] = 1.0
        sorted_frequency = np.sort(seqdict[key])
        cat = np.argmax(seqdict[key])
        y_array[-1] = (sorted_frequency[-1]+1) / (np.sum(sorted_frequency)+3)
        if cat > 0:
            y_array[0] = 1.0
        if cat > 1:
            y_array[1] = 1.0
        y_array[-2] = cat
        if sorted_frequency[-1] > sorted_frequency[-2]:
            y.append(y_array)
            x.append(x_array)
        else:
            unclear_category += 1
    x, y = np.vstack(x), np.stack(y)
    print('full dataset size: %s,%s'%(x.shape[0], x.shape[1]))
    print("%s unclear category"%unclear_category)
    np.random.seed(0)
    indices = np.random.choice(x.shape[0], x.shape[0], replace=False)
    x = torch.from_numpy(x[indices,:]).float()
    y = torch.from_numpy(y[indices,:])
    trainx = x[0:int(0.8*y.shape[0]), :]
    trainy = y[0:int(0.8*y.shape[0]),:]

    testx = x[int(0.8*y.shape[0]):, :]
    testy = y[int(0.8*y.shape[0]):, :]
    return trainx, trainy, testx, testy

#This function is a wrapper for protvec_encode_single_dataset and is
#responsible for encoding the one-hot data using the protvec representation
#(see Asgari et al for details). Basically, each kmer is assigned a numeric vector
#and these are added to produce the final representation. While this is
#significantly outperformed by simple one-hot encoding, surprisingly it outperform
#what have been claimed to be more sophisticated "language" models, hence its inclusion.
def encode_protvec(start_dir, trainx, testx):
    os.chdir(start_dir)
    os.chdir(os.path.join("results_and_resources", "ProtVec"))
    kmer_dict = dict()
    with open("protVec_100d_3grams.csv") as handle:
        _ = handle.readline()
        for line in handle:
            kmer = line.split('\t')[0]
            kmer_array = [float(z) for z in line.strip().split('\t')[1:]]
            kmer_dict[kmer] = np.asarray(kmer_array)
    os.chdir(os.path.join("..", "..", "encoded_data"))
    protvec_encode_single_dataset(trainx, "protvec_trainx.pt", kmer_dict)
    protvec_encode_single_dataset(testx, "protvec_testx.pt", kmer_dict)
    os.chdir(start_dir)

#Encodes either the training set or test set using the protvec encoding.
def protvec_encode_single_dataset(data_array, output_name, kmer_dict):
    kmer_encodings = []
    for i in range(data_array.shape[0]):
        kmer_encoding = np.zeros((100))
        seq_as_string = []
        for j in range(data_array.shape[1]):
            seq_as_string.append(aas[torch.argmax(data_array[i,j,:])])
        seq_as_string = ''.join(seq_as_string).replace('-', '')
        for j in range(len(seq_as_string)-3):
            kmer_encoding = kmer_encoding + kmer_dict[seq_as_string[j:(j+3)]]
        kmer_encodings.append(kmer_encoding)
    kmer_encodings = torch.from_numpy(np.stack(kmer_encodings)).float()
    torch.save(kmer_encodings, output_name)


def run_autoencoder(start_dir, trainx, testx, model, model_type):
    """Runs the specified autoencoder on the training and test data.

    Args:
        start_dir (str): A filepath to the project directory.
        trainx (tensor): A PyTorch one-hot encoded input tensor.
        testx (tensor): A PyTorch one-hot encoded input tensor.
        model: A model object.
        model_type (str): The type of model this data will be
            used for. Used when saving the file to assign an appropriate
            name.
    """
    encoded_trainx = model.extract_hidden_rep(trainx)
    encoded_testx = model.extract_hidden_rep(testx)
    encoded_trainx = encoded_trainx[:,29:124,:]
    encoded_testx = encoded_testx[:,29:124,:]
    os.chdir(start_dir)
    os.chdir("encoded_data")
    torch.save(encoded_trainx, "%s_trainx.pt"%model_type)
    torch.save(encoded_testx, "%s_testx.pt"%model_type)
    os.chdir(start_dir)

def get_aa_distribution(position_dict, unused_positions):
    """Gets the distribution of amino acids in RH02 and RH03.

    Args:
        position_dict (dict): A dictionary mapping input sequence positions
            to Chothia numbering.
        unused_positions (list): A list of the Chothia numbered positions
            we do not use.
    """
    mutation_array = np.ones((132,21))
    for filename in ["rh02_sequences.txt", "rh03_sequences.txt"]:
        with open(filename, 'r') as file_handle:
            for line in file_handle:
                seq, freq = line.split()[0], float(line.split()[1])
                for i, aa in enumerate(seq):
                    mutation_array[position_dict[i],
                                    aas.index(aa)] += freq
    mutation_array = mutation_array[:,0:20]
    mutation_array = mutation_array / np.sum(mutation_array, axis=1)[:,None]
    np.save("rh02_rh03_prob_distribution", mutation_array)


def sequence_encoding_wrapper(start_dir):
    """A convenience function that makes all of the different encodings / representations
    available via a single function call. It loops over multiple encoding types
    and generates them if they are not present. It does NOT generate Unirep / 
    Fair-ESM since those are generated via a separate procedure (see other
    files in this dir)."""
    os.chdir(os.path.join(start_dir, "encoded_data"))
    #This is a little clunky,but unfortunately now baked into the way the pipeline is set up.
    #Before proceeding, check to make sure the raw data has already been processed.
    fnames = os.listdir()
    for fname in ["rh01_sequences.txt", "rh02_sequences.txt", "rh03_sequences.txt"]:
        if fname not in fnames:
            print("The raw data has not been processed yet. Please process the raw data before "
                    "attempting to encode.")
            return
    os.chdir(start_dir)
    adapt_model = load_model(start_dir, "TaskAdapted_Autoencoder.ptc", "adapted")
    nonadapt_model = load_model(start_dir, "Unadapted_Autoencoder.ptc", "nonadapted")

    position_dict, unused_positions = gen_anarci_dict(start_dir)
    trainx, trainy, testx, testy = load_data(start_dir, "onehot")
    if trainx is None:
        print("Now one-hot encoding the data...")
        trainx, trainy, testx, testy = one_hot_encode(start_dir, position_dict, 
                        unused_positions)
        os.chdir("encoded_data")
        torch.save(trainx, "onehot_trainx.pt")
        torch.save(trainy, "trainy.pt")
        torch.save(testx, "onehot_testx.pt")
        torch.save(testy, "testy.pt")
        os.chdir(start_dir)
    encoded_data = load_data(start_dir, "adapted")
    if encoded_data[0] is None:
        print("Now running the adapted autoencoder...")
        run_autoencoder(start_dir, trainx, testx, adapt_model, "adapted")
    encoded_data = load_data(start_dir, "nonadapted")
    if encoded_data[0] is None:
        print("Now running the unadapted_autoencoder...")
        run_autoencoder(start_dir, trainx, testx, nonadapt_model, "nonadapted")
    encoded_data = load_data(start_dir, "protvec")
    if encoded_data[0] is None:
        print("Now generating the protvec encoding...")
        encode_protvec(start_dir, trainx, testx)
    os.chdir("encoded_data")
    if "rh02_rh03_prob_distribution.npy" not in os.listdir():
        print("Now calculating marginal aa probabilities...")
        get_aa_distribution(position_dict, unused_positions)
    print("All encodings complete except for UniRep")
    os.chdir(start_dir)


if __name__ == "__main__":
    main()
