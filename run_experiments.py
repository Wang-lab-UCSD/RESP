"""Provides a command line utility to run key experiments described in the paper."""
import argparse
import os
import sys
import subprocess
import shutil

from src.model_training.model_training_code import train_evaluate_models, train_evaluate_trastuzumab
from src.model_training.model_training_code import plot_all_scores, score_trastuzumab
from src.raw_data_processing.process_raw_reads import process_all_raw_reads
from src.sequence_encoding.encode_sequences import sequence_encoding_wrapper
from src.sequence_encoding.alternate_encoding import alternate_encoding_wrapper
from src.sequence_encoding.Unirep_encoding import unirep_encoding_wrapper
from src.simulated_annealing.run_markov_chains import run_annealing_chains
from src.simulated_annealing.run_markov_chains import analyze_annealing_results
from src.simulated_annealing.run_markov_chains import score_mutations
from src.sequence_encoding.seqs_to_fasta import convert_to_fasta
from src.sequence_encoding.fair_esm_wrapper import fair_esm_wrapper
from src.model_training.cv_scoring import run_all_5x_cvs
from src.trastuzumab_exp.process_raw_seqs import encode_trastuzumab_seqs

def gen_arg_parser():
    parser = argparse.ArgumentParser(description="Use this command line app "
            "to run / reproduce all of the key steps in the pipeline: "
            "processing raw sequence data, training and evaluating models, "
            "running simulated annealing and analyzing the simulated annealing "
            "results.\n\nSpecify which steps you would like to execute. Note that "
            "some steps must be executed before other steps can be executed. "
            "For example, setup is a necessary first step in the pipeline. You "
            "can execute one step at a time, or specify all to perform all "
            "of them in sequence.")
    parser.add_argument("--setup", action="store_true", help=
            "Download the raw sequence files from SRA. NOTE: This feature "
            "will only be available once raw reads are in SRA -- coming soon...")
    parser.add_argument("--processraw", action="store_true", help=
            "Process the raw sequences.")
    parser.add_argument("--downloadencodings", action="store_true", help=
            "Some of the embeddings described in the paper are expensive "
            "to generate (primarily fair-esm, which is computationally "
            "quite expensive). We offer the ability to download them instead "
            "using this option.")
    parser.add_argument("--encode", action="store_true", help=
            "Encode the processed raw data using the various representations "
            "described in the paper (e.g. Unirep, one-hot etc.) Be aware that for "
            "some encodings, primarily fair-esm, this may take considerable "
            "time -- downloading them is usually preferable.")
    parser.add_argument("--altprocess", action="store_true", help=
            "Process the compiled sequences into fasta files for use by other "
            "routines.")
    parser.add_argument("--traintest", action="store_true", help=
            "Train models on the test set for each encoding and model type; "
            "reproduce the test set evaluation performed for the paper.")
    parser.add_argument("--finalmodel", action="store_true", help=
            "Train a final model on the combined training and test sets "
            "(encoded using the task-adapted autoencoder)")
    parser.add_argument("--runcvs", action="store_true", help=
            "Reproduce the 5x CV evaluations from the paper. Note that "
            "this step is likely to take several hours to run.")
    parser.add_argument("--gettopseqs", action="store_true", help=
            "Get the top scoring sequences from the orgiinal dataset.")
    parser.add_argument("--simulatedanneal", action="store_true", help=
            "Reproduce the 10 simulated annealing experiments run for "
            "the paper")
    parser.add_argument("--analyzeanneal", action="store_true", help=
            "Analyze the results of the simulated annealing experiment "
            "and store the final selected sequences.")
    parser.add_argument("--mutcheck", action="store_true", help=
            "Check which mutations in the selected sequences are predicted "
            "to contribute most highly to affinity.")
    parser.add_argument("--plotscores", action="store_true", help=
            "Plot the score distribution vs experimentally determined "
            "binding category.")
    parser.add_argument("--trastsetup", action="store_true", help=
            "Set up the trastuzumab sequence data.")
    parser.add_argument("--trasteval", action="store_true", help=
            "Evaluate on train-test for the trastuzumab data.")
    parser.add_argument("--trastscore", action="store_true", help=
            "Score trastuzumab sequences selected for experimental eval.")
    return parser

def main():
    """This is the entry point for all of the steps in the pipeline;
    it uses argparse to parse command line arguments, then calls
    the relevant routines from specific subdirectories in src as
    appropriate. The easiest way to reproduce the experiments
    described in the paper is to use
    the command line arguments specified here. You can of course
    extract specific chunks of the pipeline code from src as 
    desired, but if so you will have to reconfigure them to work
    with your alternative pipeline.
    """
    #Many of the actions taken by routines in the pipeline involve
    #directories and files that have specific locations relative
    #to the start directory.
    start_dir = os.getcwd()
    parser = gen_arg_parser()
    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    if args.setup:
        pass
    if args.processraw:
        process_all_raw_reads(start_dir)
    if args.downloadencodings:
        #Data is temporarily stored here (move to permanent location soon):
        procstat = subprocess.run(["gdown", "1kmTs8XumNcUC8R4RQo90fk-8V_Yy7lmm"])
        tarfile = "encoded_data.tar.gz"
        shutil.move(tarfile, "encoded_data")
        os.chdir("encoded_data")
        for f in os.listdir():
            if not f.endswith(".tar.gz"):
                os.remove(f)
        procstat = subprocess.run(["tar", "-xzf", tarfile])
        os.chdir("encoded_data")
        for f in os.listdir():
            shutil.move(f, "..")
        os.chdir("..")
        os.rmdir("encoded_data")
        os.remove("encoded_data.tar.gz")
    if args.encode:
        sequence_encoding_wrapper(start_dir)
        convert_to_fasta(start_dir)
        unirep_encoding_wrapper(start_dir)
        fair_esm_wrapper(start_dir)
    if args.altprocess:
        alternate_encoding_wrapper(start_dir)
    if args.traintest:
        train_evaluate_models(start_dir, action_to_take="traintest_eval")
    if args.finalmodel:
        train_evaluate_models(start_dir, action_to_take="train_final_model")
    if args.gettopseqs:
        train_evaluate_models(start_dir, action_to_take="get_top_seqs")
    if args.simulatedanneal:
        run_annealing_chains(start_dir)
    if args.analyzeanneal:
        analyze_annealing_results(start_dir)
    if args.mutcheck:
        score_mutations(start_dir)
    if args.plotscores:
        plot_all_scores(start_dir)
    if args.runcvs:
        run_all_5x_cvs(start_dir)
    if args.trastsetup:
        encode_trastuzumab_seqs(start_dir)
    if args.trasteval:
        train_evaluate_trastuzumab(start_dir)
    if args.trastscore:
        score_trastuzumab(start_dir)

if __name__ == "__main__":
    main()
