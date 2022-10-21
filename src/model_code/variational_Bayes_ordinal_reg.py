'''Bayes-by-backprop NN adapted from Blundell et al 2015, designed
to perform ordinal regression where each sequence is assigned a "score"
and then classified based on the "score" as compared to threshold values.
The model can be run in MAP mode for reproducibility, which generates a MAP
prediction, or using sampling, which provides a (VERY crude, but nonetheless
useful) estimate of uncertainty.'''

#Author: Jonathan Parkinson <jlparkinson1@gmail.com>

import os, torch, numpy as np, time
import torch.nn.functional as F
from torch.autograd import Variable

#######################################################################


class FC_Layer(torch.nn.Module):
    """The FC_Layer is a single fully connected layer in the network.

    Attributes:
        mean_prior1: The mean of the prior distribution on the weights
        sigma_prior1: The width of the prior distribution on the weights
        input_dim: Expected input dimensionality
        output_dim: Expected output dimensionality
        pi2log (float): A useful constant
        pilog (float): A useful constant
        weight_means: The means of the distributions for the weights in the layer
        weight_rhos: The rho parameters used to generate the standard deviation of
            the distribution for each weight.
        bias_means: The means of the distributions for the biases in the layer
        bias_rhos: The rho parameters used to generate the standard deviation of
            the distribution for each bias term.
    """
    def __init__(self, n_in, n_out, sigma_prior1 = 1.0):
        super(FC_Layer, self).__init__()
        torch.manual_seed(123)
        self.register_buffer("sigma_prior1", torch.tensor([sigma_prior1]))
        self.register_buffer("mean_prior1", torch.tensor([0.0]))
        
        self.input_dim = n_in
        self.output_dim = n_out
        
        self.register_buffer("pi2log", torch.log(torch.tensor([2.0*3.1415927410125732])) )
        self.register_buffer("pilog", torch.log(torch.tensor([3.1415927410125732])) )

        self.weight_means = torch.nn.Parameter(torch.zeros((n_in, n_out)).uniform_(-0.1,0.1).float())
        self.weight_rhos = torch.nn.Parameter(torch.zeros((n_in, n_out)).uniform_(-3,-2).float())

        self.bias_means = torch.nn.Parameter(torch.zeros((n_out)).uniform_(-0.1,0.1).float())
        self.bias_rhos = torch.nn.Parameter(torch.zeros((n_out)).uniform_(-3,-2).float())


    def forward(self, x, sample=True, random_seed = None):
        """The forward pass. Notice this layer does not apply any activation.

        If we are sampling, we use the means and rhos to generate a sample
        from the normal distribution they describe for each weight, and this gives us
        our weight matrix -- we use the reparameterization trick so that the gradient can
        be evaluated analytically. Additionally, we evaluate the KL divergence -- the other term
        in the ELBO -- for the variational distribution of the weights from p(w), aka the 
        complexity term.
        Note that while we do use a diagonal Gaussian variational posterior as in the
        original paper, for p(w) we use a Cauchy distribution -- this is a little different from
        using a scale mixture of Gaussians as described in the original paper; it implies
        that while we expect the distribution of weights to be symmetric and for most weights
        to be close to zero, we are not surprised by a small population of much larger
        (absolute) values. It is hard to choose an informative prior for the weight distribution
        of a deep learning model and can be argued whether this or a scale mixture of Gaussians
        (as in the original paper) is a better choice. Based on evaluation on other datasets
        (not shown) we have found they offer similar performance.

        If not sampling (MAP mode), the KL divergence cannot be evaluated; rather than drawing
        samples we just use the mean of the weight & bias distributions for the forward pass.
        This is used only for predictions; obviously if we used this for training we would
        be reverting to a simple FCNN.
        """
        if random_seed is not None:
            torch.manual_seed(random_seed)
        if sample:
            weight_epsilons = Variable(self.weight_means.data.new(self.weight_means.size()).normal_())
            bias_epsilons = Variable(self.bias_means.data.new(self.bias_means.size()).normal_())
            weight_stds = self.softplus(self.weight_rhos)
            bias_stds = self.softplus(self.bias_rhos)

            weight_sample = self.weight_means + weight_epsilons*weight_stds
            bias_sample = self.bias_means + bias_epsilons*bias_stds
            
            output = torch.mm(x, weight_sample) + bias_sample
            kl_loss = -self.log_cauchy(self.mean_prior1, self.sigma_prior1, 
                                        weight_sample).sum()
            kl_loss = kl_loss - self.log_cauchy(self.mean_prior1, self.sigma_prior1,
                                            bias_sample).sum()

            kl_loss = kl_loss + self.log_gaussian(self.weight_means, weight_stds,
                                    weight_sample).sum()
            kl_loss = kl_loss + self.log_gaussian(self.bias_means, bias_stds,
                                    bias_sample).sum()
        else:
            kl_loss = 0
            output = torch.mm(x, self.weight_means) + self.bias_means
        return output, kl_loss

    def softplus(self, x):
        """Helper function for generating standard deviations from rho values."""
        return torch.log(1 + torch.exp(x))

    def log_gaussian(self, mean, sigma, x):
        """The log of a gaussian function -- all weights follow
        normal distributions."""
        return -0.5*self.pi2log - torch.log(sigma) - 0.5*(x-mean)**2 / (sigma**2)
    
    def log_cauchy(self, loc, scale, x):
        """The log of a univariate Cauchy distribution; used as a prior for the
        weight distribution."""
        return -self.pilog - torch.log(scale) - torch.log(1 + (x-loc)**2 / (scale**2))

class Output_Layer(torch.nn.Module):
    """This class is the output layer, which for ordinal regression is different in some
    respects from the fully connected layers used in the rest of the model. The 'score"
    output by the previous layer is added to fixed thresholds and returned. It was originally
    intended to make the thresholds learned parameters but initial experiments suggested
    use of fixed thresholds worked just as well (and reduced the number of parameters in the
    model); consequently while there are attributes for learned threshold means and rhos,
    these are not currently used."""
    def __init__(self, num_outputs = 2):
        super(Output_Layer, self).__init__()
        torch.manual_seed(123)
        self.register_buffer("pi2log", torch.log(torch.tensor([2.0*3.1415927410125732])) )
        fixed_thresh = [-num_outputs + 1.0]
        fixed_thresh += [fixed_thresh[0] + i * 2.0 for i in range(1, num_outputs)]
        self.register_buffer("fixed_thresh", torch.tensor(fixed_thresh))
        #Not currently used
        self.thresh_means = torch.nn.Parameter(torch.Tensor([-1.0,1.0]))
        self.thresh_rhos = torch.nn.Parameter(torch.zeros((2)).uniform_(-3,-2).float())
        self.register_buffer("thresh_prior", torch.tensor([-1.0,1.0]))
        self.register_buffer("sigma_prior1", torch.tensor([0.1]))

    def forward(self, x, sample=True):
        """Forward pass. Currently the input is added to fixed thresholds;
        the remaining code which would treat the thresholds as
        learned parameters is not currently used.
        """
        return x + self.fixed_thresh, 0
        
        if sample:
            thresh_epsilons = Variable(self.thresh_means.data.new(self.thresh_means.size()).normal_())
            thresh_stds = self.softplus(self.thresh_rhos)

            thresh_sample = self.thresh_means + thresh_epsilons*thresh_stds
            
            kl_loss = -self.log_gaussian(self.thresh_prior, self.sigma_prior1, 
                                        thresh_sample).sum()

            kl_loss = kl_loss + self.log_gaussian(self.thresh_means, thresh_stds,
                                    thresh_sample).sum()
            return x + thresh_sample, kl_loss
        else:
            return x + self.thresh_means, 0

    def softplus(self, x):
        """Helper function for converting rho into standard deviation."""
        return torch.log(1 + torch.exp(x))

    def log_gaussian(self, mean, sigma, x):
        """Helper function for the log of a gaussian distribution."""
        return -0.5*self.pi2log - torch.log(sigma) - 0.5*(x-mean)**2 / (sigma**2)


class bayes_ordinal_nn(torch.nn.Module):
    """The full model which is trained on the atezolizumab dataset."""
    def __init__(self, sigma1_prior = 1.0, input_dim=264, num_categories = 3):
        super(bayes_ordinal_nn, self).__init__()
        torch.manual_seed(123)
        self.num_outputs = num_categories - 1

        self.n1 = FC_Layer(input_dim, 30, sigma1_prior)
        self.n2 = FC_Layer(30, 30, sigma1_prior)
        self.n3 = FC_Layer(30,1, sigma1_prior)
        self.output_layer = Output_Layer(self.num_outputs)
        self.register_buffer("train_mean", torch.zeros((1,input_dim))  )
        self.register_buffer("train_std", torch.zeros((1,input_dim))  )


    def scale_data(self, x):
        """This function rescales input data for training or prediction if 
        the training set mean and std have been stored; if not, no change."""
        if torch.sum(self.train_mean) == 0:
            return x
        return (x - self.train_mean) / self.train_std

    def set_scaling_factors_and_scale(self, x):
        """Simultaneously calculates scaling values and scales the input data.
        Used on the training set only."""
        self.train_mean = torch.mean(x, dim=0).unsqueeze(0)
        self.train_std = torch.std(x, dim=0).unsqueeze(0).clip(min=1e-9)
        return self.scale_data(x)

    def get_class_weights(self, y):
        """Calculates the weight to apply to labels from each class given the class
        imbalance."""
        class_weights = np.zeros((y.size()[0]))
        classes = y[:,-2].numpy()
        n_instances = np.asarray([classes.shape[0] / np.argwhere(classes==i).shape[0] for
                        i in range(self.num_outputs + 1)])
        n_instances = n_instances / np.max(n_instances)
        for i in range(self.num_outputs + 1):
            class_weights[np.argwhere(classes==i).flatten()]=n_instances[i]
        return torch.from_numpy(class_weights).float()

    def forward(self, x, get_score = False, sample=True, random_seed = None):
        """The forward pass. Note that activation is applied here
        (rather than inside the FC layer). If we are getting the score
        only, we can skip the output layer, which merely adds the score
        to the predefined thresholds to get class label predictions. Since
        the score is only required post-training, if the score is sought,
        the complexity term (kl_loss) is not needed and is not returned.

        Args:
            x (tensor): The input data.
            get_score (bool): If True, get the score and do not bother
                with class label predictions.
            sample (bool): If True, generate multiple weight samples.
            random_seed (int): The seed for the random number generator
                to ensure reproducibility.

        Returns:
            If not get_loss:
            probs (tensor): The class probabilities for each datapoint.
            net_kl_loss: The KL divergence loss.
            Otherwise:
            x (tensor): The scores (the latent score for ordinal regression).
        """
        xback = torch.clone(x)
        x, kl_loss = self.n1(x, sample, random_seed)
        x = F.elu(x)
        x, kl_loss2 = self.n2(x, sample, random_seed)
        x = F.elu(x)
        x, kl_loss3 = self.n3(x, sample, random_seed)
        if get_score:
            return x
        output,_ = self.output_layer(x, sample)
        net_kl_loss = kl_loss + kl_loss2 + kl_loss3
        return torch.sigmoid(output), net_kl_loss

    def negloglik(self, ypred, ytrue, kl_loss, weights=None, class_weight_set = None):
        """Custom loss function. We calculate the binary cross-entropy loss for
        predictions and add in the complexity term. If weights are supplied,
        the loss values are weighted based on the weight for that datapoint and
        for that class (the class weights compensate for class imbalance).

        Args:
            ypred (tensor): The predicted classes.
            ytrue (tensor): The actual classes.
            kl_loss (tensor): The KL_divergence loss.
            weights (tensor): If not None, datapoint weights.
            class_weight_set (tensor): If not None, class weights. Must be supplied
                if weights is supplied.

        Returns:
            loss: The resulting loss values.
        """
        lossterm1 = -torch.log(torch.clamp(ypred, min=1*10**-10))*ytrue
        lossterm2 = -torch.log(torch.clamp(1-ypred, min=1*10**-10))*(1-ytrue)
        loss = torch.sum(lossterm1 + lossterm2, dim=1)
        if weights is not None:
            loss *= weights
            loss *= class_weight_set
            kl_loss = kl_loss * torch.sum(weights) / ypred.size()[0]
            kl_loss = kl_loss * torch.sum(class_weight_set) / ypred.size()[0]
        return (torch.sum(loss) + kl_loss) / ypred.shape[0]

    def trainmod(self, x, y, epochs=40, minibatch=200, track_loss = True,
                    lr=0.0025, use_weights = True, num_samples = 5,
                    scale_data = False, random_seed = 123):
        """This function trains the model represented by the class instance.
        If track_loss, all loss values are returned. Adam optimization with
        a low learning rate is used. Note that -- because of the way the data
        is encoded -- the datapoint weights are also stored in the y-tensor (in
        the last column), while the first two columns are one-hot indicators
        for > rh01, >rh02. Note that this is similar in many respects to typical
        NN training, but slower because of the need to draw multiple samples
        for the weights on each minibatch. Also note that currently
        we load all of the data into memory when training because our
        datasets are comparatively small and this is faster. If revising this
        for a larger dataset we should switch to loading minibatches from disk
        obviously.

        Args:
            x (tensor): The input sequence data encoded by the autoencoder
                (or other encoding scheme).
            y (tensor): The class labels. This is in an N x 4 tensor. The first
                two columns are class labels set up for ordinal regression.
                The last two are the class label in integer format and the
                datapoint weight.
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
        num_batches = int(y.size()[0]/minibatch)
        if scale_data == True:
            x = self.set_scaling_factors_and_scale(x)

        class_weights = self.get_class_weights(y)

        self.train()
        self.cuda()
        x.cuda()
        y.cuda()
        class_weights.cuda()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        losses = []
        for epoch in range(0, epochs):
            print("Epoch %s"%epoch)
            seed = None
            if random_seed is not None:
                torch.manual_seed(epoch%50)
            permutation = torch.randperm(x.size()[0])
            for j in range(0, x.size()[0], minibatch):

                indices = permutation[j:j+minibatch]
                x_mini, y_mini = x[indices,:].cuda(), y[indices,:].cuda()
                class_weights_mini = class_weights[indices].cuda()
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


    def extract_hidden_rep(self, x, num_samples = 5, use_MAP=False,
            random_seed = None):
        """This function generates the ordinal regression "score" (aka the hidden
        rep) for each input sequence, either in MAP mode (fully reproducible)
        or using a specified number of samples (stochastic). Used for scoring
        sequences to prioritize for experimental evaluation. MAP mode should
        be used for simulated annealing, where reproducible values for scores
        are desired. Otherwise, use_MAP can be set to False.

        Args:
            x (tensor): The encoded input data, a 2d tensor.
            num_samples (int): The number of weight samples to draw if NOT
                in MAP mode.
            use_MAP (bool): If True, we just generate the MAP prediction, which
                is fully reproducible, for each datapoint. THis is preferred if
                we just want to score a sequence. Otherwise, we use sampling
                to estimate our level of confidence around each score that is
                generated.
            random_seed (int): Set if reproducible sampling is desired.

        Returns:
            scorelist (tensor): The predicted scores.
            std: If use_MAP, 0 is returned. Otherwise, the standard deviation
                of the predicted score for each datapoint is returned instead.
        """
        with torch.no_grad():
            self.cpu()
            self.eval()
            scorelist = []
            scaled_x = self.scale_data(x)
            seed = random_seed
            if use_MAP == False:
                for i in range(num_samples):
                    if seed is not None:
                        seed = random_seed + i
                    scorelist.append(self.forward(scaled_x, get_score=True,
                                random_seed = seed))
                scorelist = torch.cat(scorelist, dim=-1)
                return torch.mean(scorelist, dim=-1), torch.std(scorelist, dim=-1)
            return self.forward(scaled_x, get_score=True, sample=False), 0

    
    def map_predict(self, x):
        """This function makes a MAP prediction for class label without performing any sampling. Unlike
        a sampling based prediction, this point value prediction is fully reproducible
        and should therefore be used for generating predictions in applications where
        reproducibility is desired.
        If (approximate) quantitation of uncertainty is desired,
        by contrast, predict or categorize should be used instead.
        The probabilities for class assignments are returned.
        """
        with torch.no_grad():
            self.eval()
            self.cpu()
            scaled_x = self.scale_data(x)
            return self.forward(scaled_x, sample=False)[0]
    
    def map_categorize(self, x):
        """This function is a simple wrapper on map_predict that converts
        its output to assigned categories for evaluating accuracy.

        Args:
            x (tensor): The input data.

        Returns:
            categories (np.ndarray): The category predictions.
        """
        scores = self.extract_hidden_rep(x, use_MAP=True)[0].flatten()
        categories = torch.zeros((x.size()[0]))
        for i in range(self.output_layer.fixed_thresh.shape[0]):
            categories[scores > self.output_layer.fixed_thresh[i]] = i + 1
        return categories.numpy()
        

    def categorize(self, x, num_samples=25, random_seed = None):
        """This function generates sampled predictions (no MAP mode) and
        assigns the input datapoints to categories. The predicted categories
        and the standard deviation of the scores are returned.

        Args:
            x (tensor): The input data.
            num_samples (int): The number of weight samples to draw. More
                samples is slower but increases the accuracy of the estimate.
            random_seed (int): A seed for reproducibility.

        Returns:
            std (np.ndarray): The standard deviation of the assigned scores.
            categories (np.ndarray): The predicted categories.
        """
        with torch.no_grad():
            self.eval()
            self.cpu()
            scores = []
            scaled_x = self.scale_data(x)
            seed = random_seed
            for i in range(num_samples):
                if seed is not None:
                    seed = random_seed + i
                scores.append(self.forward(scaled_x, get_score=True,
                            random_seed = seed))
            scores = torch.mean(torch.cat(scores, dim=-1), dim=-1).numpy()
            categories = np.zeros((x.size()[0]))
            for i in range(self.output_layer.fixed_thresh.shape[0]):
                categories[scores > self.output_layer.fixed_thresh[i]] = i + 1
            return np.std(scores), categories

    def predict(self, x, num_samples=5, large_testset=False, random_seed=None):
        """This function returns the probabilities of class assignment for each
        datapoint using sampling (not MAP) and the standard deviation of the scores
        _for each datapoint_.

        Args:
            x (tensor): The input data.
            num_samples (int): The number of weight samples to draw. Larger numbers
                improve accuracy and decrease speed.
            large_testset (bool): If True, loop over the dataset in chunks to
                minimize memory consumption.
            random_seed (int): A seed for reproducibility.

        Returns:
            classes (tensor): Class assignments.
            std_scores (tensor): The standard deviation of the score _for each datapoint._
        """
        with torch.no_grad():
            self.eval()
            self.cpu()
            scores = []
            scaled_x = self.scale_data(x)
            seed = random_seed
            for i in range(num_samples):
                if seed is not None:
                    seed = random_seed + i
                if large_testset == False:
                    scores.append(self.forward(scaled_x, get_score=True, random_seed = seed))
                else:
                    scores.append(self.large_testset_predictions(scaled_x, get_score=True,
                        random_seed = seed))
            scores = torch.cat(scores, dim=-1)
            mean_scores, std_scores = torch.mean(scores, dim=-1), torch.std(scores, dim=-1)
            probs, _ = self.output_layer(mean_scores.unsqueeze(-1))
            return torch.sigmoid(probs), std_scores
