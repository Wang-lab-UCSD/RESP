# PDL1_Bayesian_ML
This repo contains the code used to generate the results from Parkinson / Hard et al.
2021.

- [Installation](#Installation)
- [Usage](#Usage)
- [Citations](#Citations)

### Installation

This code was originally run on a Linux x86_64 machine with a GTX1070 Nvidia GPU
and has not been tested on other platforms at this time. Although it should in 
principle work on other platforms, it is primarily intended for use on Linux.

To install, first create and activate a Python virtual environment,
then run the following from your terminal:

```
git clone https://github.com/jlparkI/In_silico_directed_evolution
cd In_silico_directed_evolution
pip install -r requirements.txt
```

The most common cause of issues in our experience involves problems with the 
CUDA installation and/or the version of PyTorch. You may need to install a 
different version of PyTorch than the one we have here depending on your CUDA
version etc.

### Usage

To reproduce any of the experiments of interest from the pipeline, you need only
run the *run_experiments.py* script. If you do so without specifying any arguments, 
you'll see the following screen:




which essentially provides a menu of experiments you can run. Note that some of these
obviously must be run before others (for example, data must be encoded before a model
can be trained etc). Pretrained models are provided so that the models do not need
to be retrained unless desired.

Note that processing and encoding the raw read data is time-consuming and (for
some of the encoding types we tested) computationally expensive. We highly recommend
downloading the pre-encoded data instead.

### Citations

If using this code and/or pipeline for work intended for publication, please be
sure to cite:


