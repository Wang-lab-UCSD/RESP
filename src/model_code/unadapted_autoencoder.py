'''Unsupervised autoencoder trained only to reconstruct the input. 
This is distinct from the "adapted" autoencoder trained both to
reconstruct the input and make a prediction about it.'''

#Author: Jonathan Parkinson <jlparkinson1@gmail.com>

import torch, os, numpy as np, time
import torch.nn.functional as F


#######################################################################

#This class is used to fit an UnadaptedAutoencoder (our shorthand for
#unsupervised autoencoder that only reconstructs the input. This
#model is then used to encode the raw sequence data for the atezolizumab
#dataset to compare the performance of a model built using encodings from a
#non-task-adapted autoencoder as opposed to a task-adapted one.


class UnadaptedAutoencoder(torch.nn.Module):
    def __init__(self, random_seed=123):
        super(UnadaptedAutoencoder, self).__init__()
        torch.manual_seed(random_seed)
        #Encoder. Feed input through two convolutional layers and normalize
        #(a very shallow network but turns out to work surprisingly well)
        self.expander = torch.nn.Conv1d(in_channels=21, out_channels=40,
                            kernel_size=21, padding=10)
        self.compressor = torch.nn.Conv1d(in_channels=20, out_channels=6,
                                    kernel_size=11, padding=5)
        self.normalizer = torch.nn.BatchNorm1d(num_features=132,affine=False)

        #The decoder. This is a very small decoder module by most standards --
        #ensures the burden lies on the encoder to generate a meaningful representation
        self.final_adjust = torch.nn.Linear(3,21)


    #Forward pass. Uses "gated" activation (see Dauphin et al, 2016)
    def forward(self, x, decode = True, training=False):
        #encode
        x2 = x.transpose(-1,-2)
        x2 = self.expander(x2)
        x2 = x2[:,0:20,:]*torch.sigmoid(x2[:,20:,:])
        embed = self.compressor(x2).transpose(-1,-2)
        embed = embed[:,:,0:3]*torch.sigmoid(embed[:,:,3:])
        embed = self.normalizer(embed)
        if decode == False:
            return embed
        #decode
        aas = self.final_adjust(embed)
        aas = torch.softmax(aas, dim=-1)
        return aas

    #Custom loss function. Incorporates 1) the cross entropy loss for
    #the reconstruction. 
    def nll(self, aas_pred, x_mini):
        lossterm1 = -torch.log(aas_pred)*x_mini
        return torch.mean(torch.sum(torch.sum(lossterm1, dim=2), dim=1))


    #Trains the model by looping over and loading all pre-built minibatch files
    #in a specified target directory. This is a lot faster than one-hot encoding
    #the data on the fly, and more convenient than converting back and forth from
    #sparse tensors for this particular case -- the price is a large disk space
    #footprint. Returns a list of the loss fn stored on every 10 iterations.
    def train_model(self, epochs=5, minibatch=400, track_loss = True,
                traindir = 'position_anarci_pts', lr=0.005):
        start_dir = os.getcwd()
        os.chdir(traindir)
        file_list = [filename.split(".xmix")[0] for filename in os.listdir() if 
                    "xmix" in filename]

        self.train()
        self.cuda()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        losses = []
        for i in range(0, epochs):
            print("\n\nNew Epoch %s"%i)
            iterations = 0
            for current_file in file_list:
                print("%s   %s"%(current_file, file_list.index(current_file)))
                next_file, current_position = False, 0
                x = torch.load(current_file + ".xmix.pt")
                while(next_file==False):
                    if current_position >= (x.shape[0] -1):
                        next_file = True
                        break
                    elif (current_position + minibatch) > (x.shape[0]-2):
                        x_mini = x[current_position:,:,:].cuda()
                        current_position += minibatch
                        next_file = True
                    else:
                        x_mini = x[current_position:(current_position+
                                    minibatch),:,:].cuda()
                        current_position += minibatch

                    pred_aas = self.forward(x_mini, training=True)
                    pred_aas = pred_aas.clamp(min=1*10**-10)
                    loss = self.nll(pred_aas, x_mini)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if track_loss == True and iterations % 10 == 0:
                        print('Current loss: %s'%loss.item())
                        losses.append(loss.item())
                    iterations += 1
        os.chdir(start_dir)
        return losses


    #Encodes the input sequences (aka "generate hidden rep"). This function
    #is used to encode the raw one-hot sequence data for the atezolizumab 
    #dataset.
    def extract_hidden_rep(self, x, use_cpu=False):
        with torch.no_grad():
            self.eval()
            if use_cpu == True:
                self.cpu()
            else:
                self.cuda()
                x = x.cuda()
            replist = []
            if x.shape[0] < 1000:
                return self.forward(x, decode=False)
            reduced_shape = int(x.shape[0] / 1000)
            for i in range(reduced_shape):
                if i == reduced_shape - 1:
                    minibatch = x[(i*1000):,:,:]
                else:
                    minibatch = x[(i*1000):((i+1)*1000),:,:]
                if use_cpu == False:
                    minibatch = minibatch.cuda()
                replist.append(self.forward(minibatch, decode=False))
        return torch.cat(replist, dim=0).cpu()


    #Generates a reconstruction for the input. Used to
    #evaluate performance of the autoencoder.
    def predict(self, x, use_cpu = False):
        with torch.no_grad():
            self.eval()
            if use_cpu == True:
                self.cpu()
            else:
                x = x.cuda()
                self.cuda()
            replist = []
            reduced_shape = int(x.shape[0] / 1000)
            if x.shape[0] < 1000:
                return self.forward(x)
            for i in range(reduced_shape):
                if i == reduced_shape - 1:
                    minibatch = x[(i*1000):,:,:]
                else:
                    minibatch = x[(i*1000):((i+1)*1000),:,:]
                if use_cpu == False:
                    minibatch = minibatch.cuda()
                aas = self.forward(minibatch)
                replist.append(aas)
        return torch.cat(replist, dim=0).cpu().numpy()


    #Helper function that evaluates the reconstruction accuracy of the autoencoder.
    def reconstruct_accuracy(self, x, use_cpu = False):
        reps = self.predict(x, use_cpu)
        if use_cpu == True:
            x.cpu()
        pred_aas = np.argmax(reps, axis=-1)
        gt_aas = np.argmax(x.numpy(), axis=-1)
        mismatches, num_preds = 0, 0
        for i in range(x.shape[0]):
            mismatches += np.argwhere(gt_aas[i,:] != pred_aas[i,:]).shape[0]
            num_preds += gt_aas.shape[1]
        return 1 - mismatches / num_preds


