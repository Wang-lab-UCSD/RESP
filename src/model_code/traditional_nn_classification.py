'''A fairly generic fully-connected nominal classification NN
that performs three-class classification on the atezolizumab
dataset. Used as a baseline with which to compare the Bayes-by-backprop
ordinal regression NN.'''

#Author: Jonathan Parkinson <jlparkinson1@gmail.com>

import os, torch, numpy as np, time
import torch.nn.functional as F


#######################################################################

#This class is used to fit a fully connected neural net to perform
#three-class classification on the atezolizumab dataset; useful as
#a baseline for comparison with other techniques. We use dropout
#for regularization since this dramatically enhances performance on
#this task.

class fcnn_classifier(torch.nn.Module):
    def __init__(self,dropout = 0.3, input_dim=285):
        super(fcnn_classifier, self).__init__()
        torch.manual_seed(123)
        self.dropout = dropout
        self.n1 = torch.nn.Linear(input_dim,30)
        self.n2 = torch.nn.Linear(30,30)
        self.n3 = torch.nn.Linear(30,3)


    #Forward pass.
    def forward(self, x, training = True):
        x2 = F.elu(self.n1(x))
        if training == True:
            x2 = F.dropout(x2, p=self.dropout)
        x2 = F.elu(self.n2(x2))
        if training == True:
            x2 = F.dropout(x2, p=self.dropout)
        x2 = self.n3(x2)
        return torch.softmax(x2, dim=-1)


    #Trains a model on the input x and y arrays using Adam optimization
    #with simple multi-class cross entropy loss.
    def trainmod(self, x, y, epochs=5, minibatch=250, track_loss = True,
                    lr=0.0025, use_weights = True, class_weights=None):
        self.train()
        self.cuda()
        
        loss_fn = torch.nn.NLLLoss(reduction='none', weight=class_weights.float().cuda())
        x.cuda()
        y.cuda()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)    
        losses = []
        for i in range(0, epochs):
            print("Epoch %s"%i)
            torch.manual_seed(i%50)
            permutation = torch.randperm(x.size()[0])
            for j in range(0, x.size()[0], minibatch):
                indices = permutation[j:j+minibatch]
                x_mini, y_mini = x[indices,:].cuda(), y[indices,-2:].cuda()
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

        
    #Use the predict function to make predictions with a trained model.
    #Note that for classification, it returns both class probabilities AND
    #predicted categories.
    def predict(self, x):
        with torch.no_grad():
            self.eval()
            self.cpu()
            probs = self.forward(x, training=False).numpy()
            class_pred = np.argmax(probs, axis=1)
            return probs, class_pred
