'''The functions here run a prespecified number of simulated annealing
chains and save the results, using markov_chain_direv.py.'''

import os
import sys
import pickle
from copy import deepcopy

import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

from .markov_chain_direv import MarkovChainDE
from ..utilities.model_data_loader import load_model, gen_anarci_dict

#TODO: Move to constants file
full_wt = ('EVQLVESGGGLVQPGGSLRLSCAASGFTFSD--SWIHWVRQAPGKGLEWVAWISP--YGGSTYYADSVK'
        'GRFTISADTSKNTAYLQMNSLRAEDTAVYYCARRHWPGGF----------DYWGQGTLVTVSS')
aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
       'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']

#This function runs 10 chains using the modified simulated annealing algorithm
#and saves the results, together with plots of the scores over time.
def run_annealing_chains(start_dir):
    os.chdir(os.path.join(start_dir, "encoded_data"))
    try:
        prob_distro = np.load("rh02_rh03_prob_distribution.npy")
    except:
        print("The rh02_03 probability distribution was never generated. "
            "Please run the appropriate pipeline steps.")
        return

    os.chdir(os.path.join(start_dir, "results_and_resources", "simulated_annealing"))
    if "annealing_seqs_all_pos.txt" in os.listdir():
        print("Annealing results have already been saved.")
        os.chdir(start_dir)
        return

    marks = [MarkovChainDE(full_wt, prob_distro, start_dir,
                cdronly=False) for i in range(10)]
    os.chdir(os.path.join("results_and_resources", "simulated_annealing"))

    with open(f"markov_chain_scores.csv", "w+") as fhandle:
        fhandle.write(f"Chain,iteration,score\n")
    
    for i in range(len(marks)):
        marks[i].run_chain(3000, seed=i)
        fig, ax = marks[i].plot_scores()
        plt.savefig("Scores for chain %s.pdf"%i, format="pdf")
        plt.close()
        with open("markov_chain_scores.csv", "a") as fhandle:
            for j, score in enumerate(marks[i].scores):
                fhandle.write(f"{i},{j},{score}\n")

    num_retained = 0
    retained_seqs = set()
    with open("annealing_seqs_all_pos.txt", "w+") as outf:
        for k, chain in enumerate(marks):
            outf.write("\nChain %s\n"%k)
            score_indices = np.argwhere(np.asarray(chain.scores)>5).flatten()
            for idx in score_indices.tolist():
                if chain.seqs[idx] in retained_seqs:
                    continue
                num_retained += 1
                retained_seqs.add(chain.seqs[idx])
                outf.write(chain.seqs[idx] + ' ' + str(chain.scores[idx])
                        + '\n')
    os.chdir(start_dir)



def analyze_annealing_results(start_dir):
    """This function analyzes the results of the simulated annealing experiments
    to find final sequences for experimental evaluation. A great deal of the code
    here is dedicated to plotting the results."""

    os.chdir(os.path.join(start_dir, "encoded_data"))
    fnames = os.listdir()

    #As noted elsewhere, this is clunky but now unfortunately baked into
    #the pipeline. Check that all needed files are present before running.
    for fname in ["rh01_sequences.txt", "rh02_sequences.txt", "rh03_sequences.txt",
            "rh02_rh03_prob_distribution.npy"]:
        if fname not in fnames:
            print("The raw data has not been processed and/or the rh02_03 prob "
                    "distro was never generated. Please run the appropriate "
                    "pipeline steps.")
            return

    position_dict, _ = gen_anarci_dict(start_dir)
    seqs, scores = load_markov_chain_seqs(start_dir)
    varbayes_mod = load_model(start_dir,
            "atezolizumab_varbayes_model.ptc", model_type="BON")

    #We create a MarkovChainDE object to make use of its handy built
    #in features for encoding all of the sequences we just re-loaded --
    #not in order to run another chain.
    os.chdir("encoded_data")
    prob_distro = np.load("rh02_rh03_prob_distribution.npy")
    mark = MarkovChainDE(full_wt, prob_distro, start_dir)
    encoded_seqs = torch.cat([mark.encode_seq(seq) for seq in seqs], dim=0)

    os.chdir(start_dir)
    os.chdir(os.path.join("results_and_resources", "selected_sequences"))
    if "score_variability.txt" in os.listdir():
        with open("score_variability.txt", "r") as inpf:
            score_variability = [float(line.strip()) for line in inpf]
        score_variability = np.asarray(score_variability)
    else:
        _, score_variability = ordmod.extract_hidden_rep(encoded_all, 
                num_samples=1000, random_seed=0)
        with open("score_variability.txt", "w+") as outf:
            for score in score_variability.tolist():
                outf.write(str(score) + '\n')
        score_variability = score_variability.numpy()

    score_rsd = 100 * score_variability / scores

    #Why do we select the 50th percentile of scores with smallest rsd?
    #Why not 60th, or 40th? Ultimately deciding how many sequences to test
    #depends on time and budgetary constraints. The fewer sequences we
    #can afford to test, the tighter the constraints we should set.
    #50th percentile yielded a number of sequences in line with our goals.
    #In general, sequences with very high score reliability probably
    #cannot be categorized reliably, so large %RSD on score is a bad sign
    #for that sequence, although the exact cutoff is inevitably at least
    #partly an arbitrary decision.
    idx = np.argwhere(score_rsd < np.percentile(score_rsd, 50)).flatten()
    low_variability_scores = scores[idx]
    encoded_seqs = encoded_seqs.numpy()
    low_var_encodings = encoded_seqs[idx,:]
    low_variability_seqs = [seqs[i] for i in range(len(seqs)) if i in idx]
    
    distances = low_var_encodings[:,np.newaxis,:] - low_var_encodings[np.newaxis,:]
    distmat = np.sum(distances**2, axis=-1)
    distmat = squareform(distmat)
    z = linkage(distmat, method="median")
    fig, ax = plt.subplots(figsize=(15,15))
    _ = dendrogram(z, color_threshold=16)
    plt.xticks([])
    plt.savefig("Clustering for final sequence set")
    plt.close()

    dendrogram_data = pd.DataFrame.from_dict({"first_merged_id":z[:,0],
                "second_merged_id":z[:,1], "distances":z[:,2],
                "num_obs_in_cluster":z[:,3]})
    dendrogram_data.to_csv("dendrogram.csv", index=False)

    #We check whether the sequences we select were present in the original
    #dataset. This isn't crucial but is nice to know.
    os.chdir(start_dir)
    os.chdir("encoded_data")
    original_dataset = set()
    for fname in ["rh01_sequences.txt", "rh02_sequences.txt",
            "rh03_sequences.txt"]:
        with open(fname, "r") as fhandle:
            for line in fhandle:
                original_dataset.add(line.split()[0])

    os.chdir(start_dir)
    os.chdir(os.path.join("results_and_resources", "selected_sequences"))
    #Get the "shrunk" wild type with no gaps inserted to compare with
    #the selected sequences.
    shrunk_wt = full_wt.replace("-", "")
    #We plot the marginal distribution of mutations in each of the two
    #major clusters to characterize them more fully.
    temp_cluster_assignments = fcluster(z,30, criterion="distance")
    clust1 = [low_variability_seqs[i].replace("-", "") for i in range(len(low_variability_seqs))
            if temp_cluster_assignments[i] == 1]
    clust2 = [low_variability_seqs[i].replace("-", "") for i in range(len(low_variability_seqs))
            if temp_cluster_assignments[i] == 2]
    merged = clust1 + clust2
    key_positions = set()

    for seq in merged:
        for i in range(len(seq)):
            if seq[i] != shrunk_wt[i]:
                key_positions.add(i)

    key_positions = sorted(list(key_positions))
    clust1mat = np.zeros((20, len(key_positions)))
    clust2mat = np.zeros((20, len(key_positions)))
    for seq in clust1:
        for i, key_pos in enumerate(key_positions):
            clust1mat[aas.index(seq[key_pos]), i] += 1
    for seq in clust2:
        for i, key_pos in enumerate(key_positions):
            clust2mat[aas.index(seq[key_pos]), i] += 1
    #clust2mat[clust2mat < 1] = np.nan
    #clust1mat[clust1mat < 1] = np.nan
    axis_labels = [f"{key_pos + 1}\n({position_dict[key_pos]})" for key_pos in 
                    key_positions]
    fig, (ax1, ax2) = plt.subplots(2, figsize=(6,12), sharex=True)
    #Graph marginal distributions (for the paper)
    im1 = ax1.imshow(clust1mat, cmap="Reds", aspect=0.5)
    ax1.set_yticks(np.arange(20))
    ax1.set_yticklabels(aas[:-1])
    cbar = fig.colorbar(im1, ax=ax1)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.set_label("Number of sequences", rotation=270)

    im2 = ax2.imshow(clust2mat, cmap="Greens", aspect=0.5)
    ax2.set_yticks(np.arange(20))
    ax2.set_yticklabels(aas[:-1])
    cbar = fig.colorbar(im2, ax=ax2)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.set_label("Number of sequences", rotation=270)
    
    ax2.set_xticks(np.arange(0,len(key_positions)))
    ax2.set_xticklabels(axis_labels)
    ax2.set_xlabel("Wild type position\n(Chothia numbered position)")
    ax1.set_ylabel("Amino acid")
    ax2.set_ylabel("Amino acid")
    ax1.set_title("Cluster 1 marginal distributions")
    ax2.set_title("Cluster 2 marginal distributions")
    plt.savefig("Cluster marginal distributions.pdf", format="pdf")
    plt.close()

    with open("cluster_marginal_distributions_source_data.csv", "w+") as fhandle:
        for i, cluster_mat in enumerate([clust1mat, clust2mat]):
            fhandle.write(f"Source data for cluster {i+1} marginals plot\n,")
            for key_pos in key_positions:
                fhandle.write(f"{key_pos},")
            fhandle.write("\n")
            for j in range(cluster_mat.shape[0]):
                fhandle.write(f"\n{aas[j]},")
                for k in range(cluster_mat.shape[1]):
                    fhandle.write(f"{cluster_mat[j,k]},")
            fhandle.write("\n\n\n")

    #16 is based on inspection of the dendrogram the first time
    #this experiment was conducted -- and as discussed above, also
    #the need to get an appropriate number of sequences for experimental
    #eval.
    cluster_assignments = fcluster(z, 16, criterion="distance")
    num_clusts = np.unique(cluster_assignments).shape[0]
    selected_seqs = []
    for clustnum in range(1, num_clusts + 1):
        idx = np.where(cluster_assignments == clustnum)[0]
        best_idx = idx[np.argsort(low_variability_scores[idx])][-2:]
        #Keep the best two sequences in each cluster (or best sequence
        #if there is only one)
        selected_seqs.append( [low_variability_seqs[best_idx[0]], 
            low_variability_scores[best_idx[0]], clustnum ])
        if best_idx.shape[0] > 1:
            selected_seqs.append( [low_variability_seqs[best_idx[1]], 
                low_variability_scores[best_idx[1]], clustnum ])

    if "Final_Selected_Sequences.txt" in os.listdir():
        print("Final selected sequences file has already been created!")
        return

    fhandle = open("Final_Selected_Sequences.txt", "w+")
    for seq_data in selected_seqs:
        seq = seq_data[0].replace("-", "")
        mutated_positions = []
        for j in range(len(seq)):
            if seq[j] != shrunk_wt[j]:
                mutated_positions.append("%s%s%s"%(shrunk_wt[j], [j+1],
                    seq[j]))
        is_in_dataset = "no"
        if seq in original_dataset:
            is_in_dataset = "yes"
        #This is a little messy but basically just writes the selected sequences
        #to the output file in a standard format.
        outstring = (">score %s cluster %s   in_original_dataset %s    %s\n%s\n"%
                        (seq_data[1], seq_data[2], is_in_dataset, 
                        ",".join(mutated_positions), seq) )
        fhandle.write(outstring)

    fhandle.close()
    print("Final selected sequences written to file for experimental evaluation.")



def score_mutations(start_dir):
    """Scores individual mutations in the sequences selected for
    experimental evaluation to determine how crucial they are."""
    os.chdir(os.path.join(start_dir, "results_and_resources", "selected_sequences"))
    fnames = os.listdir()
    if "Final_Selected_Sequences.txt" not in fnames:
        print("The full pipeline has not yet been run; no sequences have "
            "been selected for experimental eval.")
        return

    final_scores, mutants = [], []
    with open("Final_Selected_Sequences.txt", "r") as fhandle:
        for line in fhandle:
            if line.startswith(">"):
                final_scores.append(float(line.split()[1]))
                mutants.append(line.strip().split()[-1])

    os.chdir(start_dir)
    position_dict, _ = gen_anarci_dict(start_dir)

    high_scoring_seqs = []
    unique_mutations = set()
    for mutant in mutants:
        mutations, positions = [], []
        for s in mutant.split(","):
            mutations.append(s.split("]")[-1])
            unique_mutations.add(s)
            positions.append(int(s.split("[")[1].split("]")[0]) - 1)
        positions = [position_dict[position] for position in positions]
        high_scoring_seq = deepcopy(list(full_wt))
        for mutation, position in zip(mutations, positions):
            high_scoring_seq[position] = mutation
        high_scoring_seq = "".join(high_scoring_seq)
        high_scoring_seqs.append( (high_scoring_seq, positions) )

    varbayes_mod = load_model(start_dir,
            "atezolizumab_varbayes_model.ptc", model_type="BON")
    #We create a MarkovChainDE object to make use of its handy built
    #in features for encoding all of the sequences we just re-loaded --
    #not in order to run another chain.
    os.chdir("encoded_data")
    prob_distro = np.load("rh02_rh03_prob_distribution.npy")
    mark = MarkovChainDE(full_wt, prob_distro, start_dir)

    #Two lists -- one to track impact of mutation if introduced into WT,
    #the other to track impact if subtracted.
    seqs_with_reverse_score_changes = []
    wt_score = mark.score_seq(full_wt)
    fwd_score_shift_dict = dict()
    for mutant, final_score, high_scoring_seq in zip(mutants, final_scores,
                    high_scoring_seqs):
        seq, positions = high_scoring_seq
        seq = list(seq)
        for mutation, position in zip(mutant.split(","), positions):
            if mutation in fwd_score_shift_dict:
                continue
            alt_seq = deepcopy(list(full_wt))
            alt_seq[position] = seq[position]
            new_score = mark.score_seq("".join(alt_seq))
            fwd_score_shift_dict[mutation] = new_score - wt_score

        score_change = []
        for position in positions:
            alt_seq = deepcopy(seq)
            alt_seq[position] = full_wt[position]
            new_score = mark.score_seq("".join(alt_seq))
            score_change.append(new_score - final_score)
        seqs_with_reverse_score_changes.append( (mutant, positions, score_change) )

    os.chdir(start_dir)
    os.chdir(os.path.join("results_and_resources", "selected_sequences"))
    fhandle = open("Mutation_scoring.csv", "w+")
    fhandle.write("Mutation,forward_score,reverse_scores\n")
    for mutation, fwd_score in fwd_score_shift_dict.items():
        fhandle.write(f"{mutation},{fwd_score},")
        for rev_data in seqs_with_reverse_score_changes:
            if mutation not in rev_data[0]:
                continue
            position = rev_data[0].split(",").index(mutation)
            underscored_mutation = "_".join(rev_data[0].split(","))
            fhandle.write(f"{underscored_mutation},{rev_data[2][position]}")
        fhandle.write("\n")
    fhandle.close()
    os.chdir(start_dir)


#A helper function to load the sequences saved from the simulated
#annealing step.
def load_markov_chain_seqs(start_dir):
    os.chdir(start_dir)
    os.chdir(os.path.join("results_and_resources", "simulated_annealing"))
    #By reading to a dict, then back into a list, we kill off
    #any surviving redundant seqs id'd by two separate chains.
    chainseqs = dict()
    with open("annealing_seqs_all_pos.txt", "r") as inpf:
        for line in inpf:
            if line.startswith("Chain") or len(line.strip()) == 0:
                continue
            chainseqs[line.split()[0]] = float(line.strip().split()[1])
    seqs = [key for key in chainseqs]
    scores = np.asarray([chainseqs[key] for key in seqs])
    os.chdir(start_dir)
    return seqs, scores
