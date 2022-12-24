'''A fairly generic fully-connected nominal classification NN
that performs three-class classification on the atezolizumab
dataset. Used as a baseline with which to compare the Bayes-by-backprop
ordinal regression NN.

This class inerits from fcnn_classifier and is the same in all
respects EXCEPT that it has been designed to work with data
too large to load to memory (e.g. antiberty).'''

import os, torch, numpy as np, time
import torch.nn.functional as F
from .traditional_nn_classification import fcnn_classifier

#######################################################################


class batched_fcnn_classifier(fcnn_classifier):
    """This class is a batched version of the fcnn classifier, designed
    to work with datasets too large to load to memory.
    """
    def __init__(self,dropout = 0.3, input_dim=285):
        super().__init__(dropout, input_dim)


    def trainmod(self, xfiles, yfiles, epochs=5, minibatch=250, track_loss = True,
                    lr=0.0025, use_weights = True, class_weights=None):
        """Trains a model on the input x and y arrays using Adam optimization with
        simple multi-class cross-entropy loss. This class has been designed to
        work with datasets too large to load to memory.

        Args:
            xfiles (list): A list of absolute filepaths to x data files.
            yfiles (list): A list of absolute filepaths to y data files.
            epochs (int): Number of epochs for training.
            minibatch (int): Minibatch size.
            track_loss (bool): Whether to track loss and return that information.
            lr (float): The learning rate for Adam.
            use_weights (bool): If True, use datapoint weighting. True by default.
            class_weights: A PyTorch array containing the class weights.

        Returns:
            losses (list): A list of the losses IF track_loss is True, otherwise,
                nothing is returned.
        """
        self.train()
        self.cuda()
        
        loss_fn = torch.nn.NLLLoss(reduction='none', weight=class_weights.float().cuda())
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        losses = []
        for i in range(0, epochs):
            print("Epoch %s"%i)
            torch.manual_seed(i%50)
            permutation = torch.randperm(len(xfiles)).tolist()
            for j in permutation:
                x_mini, y_mini = torch.load(xfiles[j]).float(),\
                        torch.load(yfiles[j]).cuda()
                x_mini = torch.flatten(x_mini, start_dim=1).cuda()
                y_pred = torch.log(self.forward(x_mini).clamp(min=1*10**-10))
                if use_weights:
                    loss = loss_fn(y_pred, y_mini[:,-2].long())*y_mini[:,-1]
                else:
                    loss = loss_fn(y_pred, y_mini[:,-2].long())
                loss = torch.mean(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if track_loss:
                    losses.append(loss.item())
            if track_loss == True:
                print('Current loss: %s'%loss.item())
        if track_loss == True:
            return losses
