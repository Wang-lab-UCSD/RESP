'''Bayes-by-backprop NN adapted from Blundell et al 2015, designed
to perform ordinal regression where each sequence is assigned a "score"
and then classified based on the "score" as compared to threshold values.
The model can be run in MAP mode for reproducibility, which generates a MAP
prediction, or using sampling, which provides a (VERY crude, but nonetheless
useful) estimate of uncertainty.

Note that this model differs ONLY from the variational Bayes ordinal
reg file in that it is designed to train on datasets too large to load
to memory (i.e. Antiberty). Otherwise it is the same. It therefore
takes as input a list of xfiles and a list of yfiles.'''

#Author: Jonathan Parkinson <jlparkinson1@gmail.com>

import os, torch, numpy as np, time
import torch.nn.functional as F
from torch.autograd import Variable
from .variational_Bayes_ordinal_reg import bayes_ordinal_nn


class batched_bayes_ordinal_nn(bayes_ordinal_nn):
    """The full model which is trained on the atezolizumab dataset.
    Note that this class inherits from bayes_ordinal_nn and is in
    most respects the same. The ONLY difference is training, where
    this model accepts a list of files as input, because it is
    intended to work with datasets too large to load into memory."""
    def __init__(self, sigma1_prior = 1.0, input_dim=264, num_categories = 3):
        super().__init__(sigma1_prior, input_dim, num_categories)
        self.class_weight_instances = None

    def set_scaling_factors_and_scale(self, xfiles):
        """Simultaneously calculates scaling values and scales the input data.
        Used on the training set only. Parent class is overwritten
        since we can only load a batch at a time, whereas parent class
        was intended for data loaded to memory."""
        x1 = torch.load(xfiles[0])
        self.train_mean = torch.zeros((1,x1.shape[0]))
        self.train_std = torch.zeros((1,x1.shape[0]))
        ndpoints = 0
        for xfile in xfiles:
            x = torch.load(xfile)
            batch_mean = torch.mean(x, dim=0)
            batch_std = torch.std(x, dim=0)

            updated_std = self.train_mean[0,:]**2 + self.train_std[0,:]**2
            updated_std *= ndpoints
            updated_std += x.shape[0] * (batch_mean**2 + batch_std**2)
            updated_std /= (ndpoints + x.shape[0])
            updated_std -= (ndpoints * self.train_mean[0,:] + x.shape[0] *
                    batch_std)**2 / (ndpoints + x.shape[0])

            self.train_std[0,:] = torch.sqrt(updated_std)
            self.train_mean[0,:] = self.train_mean[0,:] * ndpoints + \
                    torch.mean(x, dim=0) * x.shape[0]
            ndpoints += x.shape[0]
            self.train_mean /= ndpoints

    def get_class_weights(self, yfiles):
        """Calculates the weight to apply to labels from each class given the class
        imbalance. Overwrites the parent class since we need to be able
        to generate these for batches rather than the whole dataset."""
        classes = [torch.load(yfile)[:,-2].numpy() for yfile in yfiles]
        classes = np.concatenate(all_labels)
        n_instances = np.asarray([classes.shape[0] / np.argwhere(classes==i).shape[0] for
                        i in range(self.num_outputs + 1)])
        self.class_weight_instances = n_instances / np.max(n_instances)


    def get_batch_class_weights(self, y):
        """Gets class weights for a minibatch of data."""
        class_weights = np.zeros((y.size()[0])) 
        for i in range(self.num_outputs + 1):
            class_weights[np.argwhere(classes==i).flatten()] = \
                    self.class_weight_instances[i]
        return torch.from_numpy(class_weights).float()


    def trainmod(self, xfiles, yfiles, epochs=40, minibatch=200, track_loss = True,
                    lr=0.0025, use_weights = True, num_samples = 5,
                    scale_data = False, random_seed = 123):
        """This function trains the model represented by the class instance.
        If track_loss, all loss values are returned. Adam optimization with
        a low learning rate is used. Note that -- because of the way the data
        is encoded -- the datapoint weights are also stored in the y-tensor (in
        the last column), while the first two columns are one-hot indicators
        for > rh01, >rh02. Note that this is similar in many respects to typical
        NN training, but slower because of the need to draw multiple samples
        for the weights on each minibatch.

        The only difference between this function and the function for the parent
        class (which it overwrites) is that this one takes lists of files as input
        rather than data loaded to memory. This is necessary for larger datasets
        & encodings (e.g. antiberty).

        Args:
            xfiles (list): A list of x data files (all should be pytorch tensors saved
                as pt).
            yfiles (list): The data files containing class labels. Each is in an N x 4
                tensor. The first two columns are class labels set up for
                ordinal regression. The last two are the class label in integer
                format and the datapoint weight.
            epochs (int): The number of training epochs.
            minibatch (int): The minibatch size.
            track_loss (bool): If True, track loss values during training and
                return them as a list.
            lr (float): The learning rate for Adam.
            use_weights (bool): If True, use datapoint weighting. Defaults to True.
            num_samples (int): The number of weight samples to draw on each
                minibatch pass. A larger number will speed up convergence for
                training but make it more expensive. Defaults to 5.
            scale_data (bool): If True, store the dataset mean and variance
                and use this to scale both the training data and any future
                test data. Defaults to False.
            random_seed: If not None, should be an integer seed for the random
                number generator.

        Returns:
            losses (list): A list of loss values.
        """
        num_batches = len(yfiles)
        if scale_data == True:
            self.set_scaling_factors_and_scale(xfiles)

        class_weights = self.get_class_weights(yfiles)

        self.train()
        self.cuda()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        losses = []
        for epoch in range(0, epochs):
            print("Epoch %s"%epoch)
            seed = None
            if random_seed is not None:
                torch.manual_seed(epoch%50)
            permutation = torch.randperm(len(xfiles)).tolist()
            for idx in permutation:
                x_mini, y_mini = torch.load(xfiles[j]).float(),\
                        torch.load(yfiles[j])
            for j in range(0, x.size()[0], minibatch):
                indices = permutation[j:j+minibatch]
                x_mini, y_mini = x[indices,:].cuda(), y[indices,:].cuda()
                class_weights_mini = self.get_batch_class_weights(y_mini).cuda()
                loss = 0
                for i in range(num_samples):
                    if random_seed is not None:
                        seed = random_seed + i + j + epoch
                    y_pred, kl_loss = self.forward(x_mini, 
                                random_seed = seed)
                    y_pred = y_pred.clamp(min=1e-10)
                    if use_weights:
                        loss += self.negloglik(y_pred, y_mini[:,:self.num_outputs], kl_loss/num_batches,
                                    y_mini[:,-1], class_weights_mini)
                    else:
                        loss += self.negloglik(y_pred, y_mini, kl_loss/num_batches)
                loss = loss / num_samples
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if track_loss:
                    losses.append(loss.item())
            if track_loss == True:
                print('Current loss: %s'%loss.item())
        if track_loss == True:
            return losses
