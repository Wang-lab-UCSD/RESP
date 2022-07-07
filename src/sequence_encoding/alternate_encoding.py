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

import os, numpy as np, Bio, torch, sys, random
from Bio.Seq import Seq
from ..model_code.task_adapted_autoencoder import TaskAdaptedAutoencoder as TAE
from ..model_code.unadapted_autoencoder import UnadaptedAutoencoder as UAE
from ..utilities.model_data_loader import gen_anarci_dict, load_data, load_model

aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
       'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']


wildtype = ('EVQLVESGGGLVQPGGSLRLSCAASGFTFSDSWIHWVRQAPGKGLE'
            'WVAWISPYGGSTYYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCARRHWPGGFDYWGQGTLVTVSS')

def alternate_processing(start_dir, position_dict, unused_positions):
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
    os.chdir("encoded_data")
    x, y = [], []
    for key in seqdict:
        x_array = ["-" for i in range(132)]
        for j, letter in enumerate(key):
            x_array[position_dict[j]] = letter
        freq_counts = seqdict[key]
        sorted_counts = np.sort(freq_counts)
        if sorted_counts[-1] == sorted_counts[-2]:
            unclear_category += 1
            continue
        out_seq = "".join(x_array)
        for i, freq_count in enumerate(freq_counts.tolist()):
            for j in range(int(freq_count)):
                x.append(out_seq)
                y.append(i)
    print('full dataset size: %s'%(len(x)))
    print(f"{unclear_category} unclear category")
    random.seed(0)
    idx = list(range(len(x)))
    random.shuffle(idx)
    x = [x[i] for i in idx]
    y = [y[i] for i in idx]
    cutoff = int(0.8 * len(y))
    trainx = x[0:cutoff]
    trainy = y[0:cutoff]

    testx = x[cutoff:]
    testy = y[cutoff:]
    with open("all_training_combined.faa", "w+") as fhandle:
        for x, y in zip(trainx, trainy):
            fhandle.write(f">Category_{y}\n")
            fhandle.write(f"{x}\n")
    with open("all_test_combined.faa", "w+") as fhandle:
        for x, y in zip(testx, testy):
            fhandle.write(f">Category_{y}\n")
            fhandle.write(f"{x}\n")



#A convenience function that makes all of the different encodings / representations
#available via a single function call.
def alternate_encoding_wrapper(start_dir):
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

    position_dict, unused_positions = gen_anarci_dict(start_dir)
    print("Now compiling fasta files...")
    alternate_processing(start_dir, position_dict, unused_positions)
    os.chdir(start_dir)


