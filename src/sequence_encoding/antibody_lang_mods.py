"""Provides encoding for the ABLang and AntiBerty models. These
are handled separately since they require additional conda dependencies
not used in the original pipeline to encode the sequences (training
and testing, by contrast, is not changed and is through the main
pipeline)."""

#Author: Jonathan Parkinson <jlparkinson1@gmail.com>

import time
import os
import sys
import numpy as np
import Bio
import torch
from Bio.Seq import Seq
import ablang
from .encode_sequences import one_hot_encode
from ..utilities.model_data_loader import gen_anarci_dict, load_data, load_model
from igfold import IgFoldRunner

#TODO: Move aas and wildtype to a constants file.
aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
       'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']
wildtype = ('EVQLVESGGGLVQPGGSLRLSCAASGFTFSDSWIHWVRQAPGKGLE'
            'WVAWISPYGGSTYYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCARRHWPGGFDYWGQGTLVTVSS')


def antiberty_encode(start_dir, full = False):
    """Encodes the sequences using the AntiBertY model."""
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

    trainx, trainy, testx, testy = one_hot_encode(start_dir, position_dict, unused_positions,
            return_unencoded_seqs = True)
    if full:
        antiberty_embed_seqgroup(trainx, trainy, start_dir, "train")
        antiberty_embed_seqgroup(testx, testy, start_dir, "test")
    else:
        antiberty_embed_avg_seqgroup(trainx, trainy, start_dir, "train")
        antiberty_embed_avg_seqgroup(testx, testy, start_dir, "test")
    os.chdir(start_dir)


def antiberty_embed_avg_seqgroup(seqs, ydata, start_dir, output_name):
    igfold = IgFoldRunner()
    sequences = {"H":seqs[0]}
    os.chdir(os.path.join(start_dir, "encoded_data"))

    embeds, ysubset = [], []
    for i, seq in enumerate(seqs):
        sequences["H"] = seq
        with torch.no_grad():
            embed = igfold.embed(sequences = sequences).bert_embs.detach().cpu()
        embeds.append(torch.mean(embed, dim=1))
        ysubset.append(ydata[i:i+1,:])
        torch.cuda.empty_cache()
        if i % 5000 == 0:
            print(f"{i} complete.\n\n\n")
    embeds = torch.cat(embeds, dim=0)
    torch.save(embeds, f"antiberty_{output_name}x.pt")
    torch.save(ysubset, f"antiberty_{output_name}y.pt")




def antiberty_embed_seqgroup(seqs, ydata, start_dir, output_name):
    igfold = IgFoldRunner()
    sequences = {"H":seqs[0]}
    os.chdir(os.path.join(start_dir, "encoded_data"))
    if "antiberty_embeds" not in os.listdir():
        os.mkdir("antiberty_embeds")
    os.chdir("antiberty_embeds")

    batchnum, embeds, ysubset = 0, [], []
    for i, seq in enumerate(seqs):
        sequences["H"] = seq
        with torch.no_grad():
            embeds.append(igfold.embed(sequences = sequences).bert_embs.detach().cpu())
        ysubset.append(ydata[i:i+1,:])
        torch.cuda.empty_cache()
        if len(embeds) >= 200:
            embeds, ysubset = save_antiberty_batch(embeds, ysubset,
                    batchnum, output_name)
            batchnum += 1
            #time.sleep(0.2)
        if i % 5000 == 0:
            print(f"{i} complete.\n\n\n")
    if len(embeds) > 0:
        embeds, ysubset = save_antiberty_batch(embeds, ysubset,
                    batchnum, output_name)


def save_antiberty_batch(embeds, ysubset, batchnum, output_name):
    embeds = torch.cat(embeds, dim=0)
    ysubset = torch.cat(ysubset, dim=0)
    torch.save(embeds, f"antiberty_{output_name}_{batchnum}_x.pt")
    torch.save(ysubset, f"antiberty_{output_name}_{batchnum}_y.pt")
    return [], []

def ablang_encode(start_dir):
    """Encodes the sequences using the ABLang model."""
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

    trainx, trainy, testx, testy = one_hot_encode(start_dir, position_dict, unused_positions,
            return_unencoded_seqs = True)

    heavy_ablang = ablang.pretrained("heavy", device="cuda")
    heavy_ablang.freeze()

    np_trainx = heavy_ablang(trainx, mode="seqcoding")
    np_testx = heavy_ablang(testx, mode="seqcoding")
    trainx = torch.from_numpy(np_trainx)
    testx = torch.from_numpy(np_testx)
    os.chdir(os.path.join(start_dir, "encoded_data"))
    torch.save(trainx, "ablang_trainx.pt")
    torch.save(testx, "ablang_testx.pt")
