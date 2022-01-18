#In contrast to the other encodings, the Unirep and fair-esm
#procedures require fasta files as input and do not use Chothia
#numbering. This function converts the processed raw data to
#fasta files that can be used by fair-esm and by TAPE (TAPE
#generates the Unirep encoding).


import os, numpy as np, Bio, torch
from Bio.Seq import Seq


aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
       'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']


wildtype = ('EVQLVESGGGLVQPGGSLRLSCAASGFTFSDSWIHWVRQAPGKGLE'
            'WVAWISPYGGSTYYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCARRHWPGGFDYWGQGTLVTVSS')

#First we assign classes and weights (same process as for all other
#encodings) which contain the labels, and we write the raw
#aa sequences to fasta files with a specific format required
#by TAPE and by fair-esm. These FASTA files will then be converted to 
#npz files using subprocess calls to TAPE, and those in turn
#are converted to PyTorch tensors. the fair-esm procedure,
#by contrast, converts directly to pytorch tensors.
def convert_to_fasta(start_dir):
    os.chdir(os.path.join(start_dir, "encoded_data"))
    if "fair_unirep_train.faa" in os.listdir() and "fair_unirep_test.faa" in os.listdir():
        print("Fasta files already generated.")
        os.chdir(start_dir)
        return
    print("Now generating fasta files for Unirep and fair-esm encoding.")
    seqdict, total_seqs = dict(), 0
    filelist = ["rh01_sequences.txt", "rh02_sequences.txt",
                    "rh03_sequences.txt"]
    for k, filename in enumerate(filelist):
        if filename not in os.listdir():
            print("Raw data has not been processed yet. Please process the "
                    "raw data before proceeding.")
            os.chdir(start_dir)
            return
        file_handle = open(filename)
        for curr_line in file_handle:
            full_seq_aa = curr_line.split('\t',1)[0]
            frequency = int(curr_line.split('\t', 1)[1].strip())
            if full_seq_aa == wildtype:
                continue
            if full_seq_aa[0:29] != wildtype[0:29] or full_seq_aa[-8:] != wildtype[-8:]:
                continue
            if full_seq_aa not in seqdict:
                seqdict[full_seq_aa] = np.zeros((len(filelist)))
            seqdict[full_seq_aa][k] = frequency
        file_handle.close()
    print('Total seqs found: %s'%(len(seqdict)))

    x, y = [], []
    for key in seqdict:
        y_array = np.zeros((len(filelist)+1))
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
            x.append(key)

    y = np.stack(y)
    print('full dataset size: %s'%(y.shape[0]))
    np.random.seed(0)
    indices = np.random.choice(y.shape[0], y.shape[0], replace=False)
    x = [x[i] for i in indices.tolist()]
    y = torch.from_numpy(y[indices,:])

    cutpoint = int(0.8 * y.shape[0])
    trainy = y[0:cutpoint,:]
    testy = y[cutpoint:, :]
    trainx = x[0:cutpoint]
    testx = x[cutpoint:]

    with open("fair_unirep_train.faa", "w+") as out_handle:
        for i, seq in enumerate(trainx):
            header_arr = '_'.join([str(i)] + [str(z) for z 
                in trainy[i,:].tolist()])
            out_handle.write(">%s\n%s\n"%(header_arr, seq))
    with open("fair_unirep_test.faa", "w+") as out_handle:
        for i, seq in enumerate(testx):
            header_arr = '_'.join([str(i)] + [str(z) for z 
                in testy[i,:].tolist()])
            out_handle.write(">%s\n%s\n"%(header_arr, seq))
    os.chdir(start_dir)
