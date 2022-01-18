'''This class performs modified simulated annealing, where
sequences are generated through random modifications to the wild-type,
with the probability of a given mutation determined by the abundance
of that aa at that position in the training set. The "score" from
ordinal regression is then contrasted with the score of the premodification
sequence and the modification is accepted with a probability determined
by the temperature. To limit the size of the search space, which would
otherwise be unmanageably large, the top ten positions at which mutations
in high-scoring sequences were most frequently observed are used.
The accepted sequences from the chain and their scores are saved. Running
several chains yields a set of sequences that can be merged for experimental
evaluation.'''

#Author: Jonathan Parkinson <jlparkinson1@gmail.com>

import torch, random, numpy as np, os, sys, matplotlib.pyplot as plt
from ..utilities.model_data_loader import load_model
aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
       'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']

aas_no_blank = aas[:-1]
#If MAX_NUM_FAILURES, i.e. if this number of sequences are rejected in
#a row, we terminate annealing (to avoid getting stuck in an endless loop
#of proposing and rejecting sequences). Otherwise annealing is terminated when
#temperature reaches the predefined threshold.
MAX_NUM_FAILURES = 11

#This class contains attributes and methods needed to perform modified
#simulated annealing. Inputs include the final trained model, the
#autoencoder, the wild type sequence and the probability distribution
#(derived from abundance of mutations at each position in the training
#set). Note that it can be run in CDRONLY mode or not. Originally we had
#discussed exploring sequences that had mutations in the CDRs only
#but eventually decided against this; the class was set up to allow for
#both (in case we decided to do that). Use a random seed for reproducibility.
class MarkovChainDE():
    def __init__(self, full_wt_seq, prob_distro,
            start_dir, cdronly = False, seed = 0):
        self.ordinal_mod = load_model(start_dir, "atezolizumab_varbayes_model.ptc", 
                "BON")
        self.autoencoder = load_model(start_dir, "TaskAdapted_Autoencoder.ptc",
                "adapted")
        self.scores = []
        self.seqs = []
        self.acceptance_rate = 0
        self.full_wt = full_wt_seq
        if cdronly:
            self.positions_for_mutation = [29,30,49,50,52,61,100,101,105,108]
        else:
            self.positions_for_mutation = [41,44,61,73,80,82,88,100,101,123]
        self.prob_distro = prob_distro
        self.seed = seed

    #Encodes a proposed mutant seq as one-hot then converts it using the
    #autoencoder and extracts the relevant region.
    def encode_seq(self, seq):
        encoder_arr = torch.zeros((1,132,21))
        for j, letter in enumerate(seq):
            encoder_arr[0,j,aas.index(letter)] = 1.0
        encoded_rep = self.autoencoder.extract_hidden_rep(encoder_arr.cuda()).cpu()
        return encoded_rep[:,29:124,:].flatten(1,-1)

    #Scores a sequence -- either using MAP mode, which is reproducible,
    #or using sampling, which is not (if the seed changed, the score
    #would change slightly). Using MAP is preferable.
    def score_seq(self, seq, use_MAP=True, num_samples=100):
        encoded_seq = self.encode_seq(seq)
        if use_MAP == False:
            score, stdev = self.ordinal_mod.extract_hidden_rep(encoded_seq, num_samples=num_samples)
            return score.numpy()
        else:
            score, _ = self.ordinal_mod.extract_hidden_rep(encoded_seq, use_MAP=True)
            return score.flatten().numpy()[0]


    #Simple helper function for plotting scores from a completed chain.
    def plot_scores(self):
        fig, ax = plt.subplots(1)
        ax.plot(self.scores, linewidth=2.0)
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Model score")
        ax.set_title("Scores for Markov chain directed evolution")
        return fig, ax

    #This function runs a simulated annealing experiment as described above.
    def run_chain(self, max_iterations=3000, seed = 0):
        num_accepted = 0
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        current_seq = list(self.full_wt)
        current_score = self.score_seq(self.full_wt, use_MAP=True)

        i, temp, num_failures = 0, 25.0, 0
        while temp >= 0.01:
            chosen_one = np.random.choice(self.positions_for_mutation, size=1)[0]
            proposed_seq = current_seq.copy()
            new_aa = aas[np.random.choice(20, size=1,
                                p=self.prob_distro[chosen_one,:])[0]]
            proposed_seq[chosen_one] = new_aa
            
            proposed_score = self.score_seq(proposed_seq, use_MAP=True)
            acceptance_prob = np.exp( -(current_score - proposed_score) / temp)
            runif = np.random.uniform()
            if acceptance_prob > runif:
                current_score, current_seq = proposed_score, proposed_seq
                num_accepted += 1
                num_failures = 0
            else:
                num_failures += 1
            self.scores.append(current_score)
            self.seqs.append(''.join(current_seq))
            temp = temp*0.99
            i += 1
            if i % 250 == 0:
                print("Temperature: %s"%temp)
            if i > max_iterations or num_failures > 11:
                print("Num consecutive failures: %s"%num_failures)
                print("Num iterations: %s"%i)
                break
        self.acceptance_rate = num_accepted / max_iterations

    #This function is available as an option to do
    #additional processing on sequences
    #selected by the simulated annealing process. Ideally we would like
    #to use a sequence with a small number of mutations from the original.
    #The polish function takes each mutant and tries to return the amino acid
    #at as many mutated positions as possible to the aa present in the wild
    #type at that position, while keeping the score within 90% of the
    #score for the original selected mutant. We did not ultimately
    #make use of this routine however for our analysis.
    def polish(self, orig_mutant):
        current_score = self.score_seq(orig_mutant, use_MAP=True)[0]
        score_thresh = np.max([0,0.9*current_score])
        revised_mutant = list(orig_mutant)
        keep_cycle = True
        while keep_cycle:
            score_tracker = []
            proposed_mut = revised_mutant.copy()
            for position in self.positions_for_mutation:
                if proposed_mut[position] != self.full_wt[position]:
                    proposed_mut[position] = self.full_wt[position]
                    score_tracker.append(self.score_seq(proposed_mut)[0])
                else:
                    score_tracker.append(0)
            if np.max(score_tracker) < score_thresh:
                keep_cycle = False
                break
            index_of_best_mut = np.argmax(score_tracker)
            best_mut = self.positions_for_mutation[index_of_best_mut]
            revised_mutant[best_mut] = self.full_wt[best_mut]
        return ''.join(revised_mutant), self.score_seq(revised_mutant)
