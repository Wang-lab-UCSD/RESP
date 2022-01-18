#Simple script for processing the raw reads -- checking for any that are problematic,
#merging the left and right reads and counting frequencies. It takes a single argument:
#the path to a target directory.

#Author: Jonathan Parkinson <jlparkinson1@gmail.com>

import Bio, gzip, sys, os, numpy as np
from Bio import Seq
from Bio.Seq import Seq
from Bio import SeqIO



#Helper function which determines whether the sequences match in the overlap
#region and if so merges them.
def adjust_and_merge(leftseq, rightseq):
    if compare_sequences(leftseq, rightseq):
        merged_seq = ''.join([leftseq[0:36], rightseq])
        if ('*' not in merged_seq and merged_seq.startswith('EVQ') 
                and merged_seq.endswith('VSS')):
            return merged_seq
    return None

#Determines whether the left and right reads match in the overlap region,
#defined as 36:len(leftseq) on the left and 0:47 on the right (just based
#on the read size and the way they were acquired).
def compare_sequences(leftseq, rightseq):
    if leftseq[36:] == rightseq[0:47]:
        return True
    return False

def process_all_raw_reads(start_dir):
    os.chdir(start_dir)
    os.chdir("raw_data")
    fnames = os.listdir()
    if "rh01" not in fnames or "rh02" not in fnames or "rh03" not in fnames:
        print("One or more of the rh01, rh02, rh03 directories under raw_data is missing")
        return
    for target_dir in ["rh01", "rh02", "rh03"]:
        print("Now processing raw data for %s"%target_dir)
        os.chdir(start_dir)
        os.chdir(os.path.join("raw_data", target_dir))
        errcode = process_single_raw_dataset(start_dir, target_dir)
        if errcode is not None:
            os.chdir(start_dir)
            return
        print("Raw data for %s complete\n"%target_dir)

def process_single_raw_dataset(start_dir, category_filename):
    fastq_filenames = [x for x in os.listdir() if x.endswith('.fastq.gz')]
    if len(fastq_filenames) != 2:
        print("One of the raw data files is missing. Please run --setup before proceeding.")
        return "err"
    if 'R2' in fastq_filenames[0]:
        fastq_filenames = [fastq_filenames[1], fastq_filenames[0]]
    with gzip.open(fastq_filenames[0],'rt') as leftreadfile:
        leftreads = [record for record in SeqIO.parse(leftreadfile, "fastq")]
    with gzip.open(fastq_filenames[1], 'rt') as rightreadfile:
        rightreads = [record for record in SeqIO.parse(rightreadfile, "fastq")]

    print('length of list from leftreadfile: %s'%len(leftreads))
    print('length of list from rightreadfile: %s'%len(rightreads))
    
    #The accepted seqs dict tracks unique sequences and the frequency with which
    #each occurred. We also track several kinds of outcomes -- sequences that
    #are excluded because they are low quality, or are excluded because
    #they could not be merged.
    accepted_seqs = dict()
    totseqs, lowqual, nonmerged = 0, 0, 0
    nonmerged_file = open("nonmerged.txt", "w+")
    for i in range(0, len(leftreads)):
        leftphred = np.asarray(leftreads[i].letter_annotations['phred_quality'])
        rightphred = np.asarray(rightreads[i].letter_annotations['phred_quality'])
        #10 is a generous low bar for the phredscore; we are essentially excluding
        #reads where we have little or no confidence in one of the letters.
        if np.min(leftphred) > 10 and np.min(rightphred) > 10:
        #We have to trim some specific positions off of each read (again, just
        #based on the way the sequencing was set up).
            leftseq = str(leftreads[i].seq[:-2].translate())
            rightseq = str(rightreads[i].seq.reverse_complement()[2:].translate()[:-1])
            merged_seq = adjust_and_merge(leftseq, rightseq)
            #If the left and right reads do not match, adjust_and_merge returns None.
            if merged_seq is not None:
                totseqs += 1
                if merged_seq not in accepted_seqs:
                    accepted_seqs[merged_seq] = 1
                else:
                    accepted_seqs[merged_seq] += 1
            #We write nonmerged reads to a separate file for later inspection
            #to ensure we are not rejecting these in error. Review of the
            #nonmerged reads from this project indicates that
            else:
                nonmerged += 1
                nonmerged_file.write(leftseq)
                nonmerged_file.write(" ")
                nonmerged_file.write(rightseq)
                nonmerged_file.write("\n")
        else:
            lowqual += 1
        if i % 10000 == 0:
            print("%s complete"%i)
    nonmerged_file.close()

    print('%s nonmerged'%(nonmerged))
    print('%s lowqual'%lowqual)
    print('%s total acceptable sequences found'%totseqs)
    print('%s unique sequences found'%len(accepted_seqs))
    os.chdir(start_dir)
    os.chdir("encoded_data")
    with open('%s_sequences.txt'%category_filename, 'w+') as handle:
        for i, key in enumerate(accepted_seqs):
            handle.write(key + '\t' + str(accepted_seqs[key]) + '\n')
    os.chdir(start_dir)
    return None
