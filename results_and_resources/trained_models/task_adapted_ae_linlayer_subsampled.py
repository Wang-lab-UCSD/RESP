import random
import torch, os, numpy as np, time
import torch.nn.functional as F


class TaskAdaptedAeLinlayerSampled(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.expander = torch.nn.Conv1d(in_channels=21, out_channels=40,
                            kernel_size=21, padding=10)
        self.compressor = torch.nn.Conv1d(in_channels=20, out_channels=6,
                                    kernel_size=11, padding=5)
        self.normalizer = torch.nn.BatchNorm1d(num_features=132,affine=False)
        
        self.final_adjust = torch.nn.Linear(3,21)

        self.predictor = torch.nn.Linear(396,1)


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

        pred_cat = torch.sigmoid(self.predictor(embed.reshape(embed.shape[0],
                        embed.shape[1]*embed.shape[2]))).squeeze(-1)

        return aas, pred_cat


    def nll(self, aas_pred, cat_pred, x_mini, y_mini):
        lossterm1 = -torch.log(aas_pred)*x_mini
        loss = torch.mean(torch.sum(torch.sum(lossterm1, dim=2), dim=1))
        lossterm3 = torch.mean(3*F.binary_cross_entropy(cat_pred, y_mini))
        return lossterm3 + loss



    def train_model(self, epochs=5, minibatch=400, track_loss = True,
                traindir = 'position_anarci_pts', lr=0.005,
                subsample_frac = 1.0):
        start_dir = os.getcwd()
        os.chdir(traindir)
        file_list = [filename.split(".xmix")[0] for filename in os.listdir() if 
                    "xmix" in filename]
        if subsample_frac < 1:
            l_size = int(subsample_frac * len(file_list))
            random.seed(123)
            random.shuffle(file_list)
            file_list = file_list[:l_size]
            print(f"{l_size} files accepted.", flush=True)
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
                y = torch.load(current_file + ".ymix.pt").float()
                while(next_file==False):
                    if current_position >= (x.shape[0] -1):
                        next_file = True
                        break
                    elif (current_position + minibatch) > (x.shape[0]-2):
                        x_mini = x[current_position:,:,:].cuda()
                        y_mini = y[current_position:].cuda()
                        current_position += minibatch
                        next_file = True
                    else:
                        x_mini = x[current_position:(current_position+
                                    minibatch),:,:].cuda()
                        y_mini = y[current_position:(current_position+
                                    minibatch)].cuda()
                        current_position += minibatch

                    pred_aas, pred_cat = self.forward(x_mini, training=True)
                    pred_aas = pred_aas.clamp(min=1*10**-10)
                    pred_cat = pred_cat.clamp(min=1*10**-10)
                    loss = self.nll(pred_aas, pred_cat, x_mini, y_mini)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if track_loss == True and iterations % 10 == 0:
                        print('Current loss: %s'%loss.item())
                        losses.append(loss.item())
                    time.sleep(0.05)
                    iterations += 1
                time.sleep(60)
            time.sleep(60)
        os.chdir(start_dir)
        return losses

        
    def extract_hidden_rep(self, x, use_cpu=False):
        with torch.no_grad():
            self.eval()
            if use_cpu == True:
                self.cpu()
            else:
                self.cuda()
                x = x.cuda()
            return self.forward(x, decode=False).cpu()


    def predict(self, x, use_cpu = False):
        with torch.no_grad():
            self.eval()
            if use_cpu == True:
                self.cpu()
            else:
                x = x.cuda()
                self.cuda()
            aas, cat_pred = self.forward(x)
            return aas.cpu().numpy(), cat_pred.cpu().numpy()


    def reconstruct_accuracy(self, x, use_cpu = False):
        reps, _ = self.predict(x, use_cpu)
        if use_cpu == True:
            x.cpu()
        pred_aas = np.argmax(reps, axis=-1)
        gt_aas = np.argmax(x.numpy(), axis=-1)
        mismatches, num_preds = 0, 0
        for i in range(x.shape[0]):
            mismatches += np.argwhere(gt_aas[i,:] != pred_aas[i,:]).shape[0]
            num_preds += gt_aas.shape[1]
        return 1 - mismatches / num_preds

    def cat_accuracy(self, x, y, use_cpu = False):
        _, cat_preds = self.predict(x, use_cpu)
        return 1 - np.sum(np.abs(y.numpy() - cat_preds)) / y.shape[0]
