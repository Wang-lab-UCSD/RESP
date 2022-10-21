"""Provides a command line utility to run additional experiments involving
alternate antibody representations (e.g. ABLang.). These are handled
separately since they require dependencies not used in the original
set of experiments that needed to be run using Conda, whereas the
original experiments all used a venv, and the requirements file
is structured accordingly. In order to run these, you will need
to install ABLang and IgFold."""
import argparse
import os
import sys
import subprocess
import shutil

from src.sequence_encoding.antibody_lang_mods import ablang_encode
from src.sequence_encoding.antibody_lang_mods import antiberty_encode

def gen_arg_parser():
    parser = argparse.ArgumentParser(description="Use this command line app "
            "to run / reproduce additional experiments (e.g. ABLang encoding).")
    parser.add_argument("--ablangencode", action="store_true", help=
            "Encode the raw data using ABLang.")
    parser.add_argument("--antibertyencode", action="store_true", help=
            "Encode the raw data using Antiberty.")
    parser.add_argument("--antibertyfull", action="store_true", help=
            "Encode the raw data using Antiberty, but without averaging "
            "over tokens.")
    return parser

def main():
    """This is the entry point for all of the steps in the pipeline;
    it uses argparse to parse command line arguments, then calls
    the relevant routines from specific subdirectories in src as
    appropriate. 
    """
    #Many of the actions taken by routines in the pipeline involve
    #directories and files that have specific locations relative
    #to the start directory.
    start_dir = os.path.dirname(os.path.abspath(__file__))
    parser = gen_arg_parser()
    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    if args.ablangencode:
        ablang_encode(start_dir)
    if args.antibertyencode:
        antiberty_encode(start_dir)
    if args.antibertyfull:
        antiberty_encode(start_dir, full = True)

if __name__ == "__main__":
    main()
