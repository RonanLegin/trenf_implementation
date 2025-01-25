import os
import torch
import random
import numpy as np
from importlib.machinery import SourceFileLoader
from torch.utils.data import TensorDataset, DataLoader

import torch.optim as optim
from model import InvariantFlowModel

import argparse
import time
import matplotlib.pyplot as plt
seed = 786

torch.autograd.set_detect_anomaly(True)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cpu") #torch.device("cuda" if torch.cuda.is_available() else "cpu")




def main(args):
    fName = time.strftime("%Y%m%d_%H_%M")
    if not os.path.exists("saves/"):
        os.makedirs("saves/")

    p = SourceFileLoader('cf', 'config_kappa.py').load_module()
    model = InvariantFlowModel(image_shape=p.imShape, n_layers=p.n_layers, learn_top=p.y_learn_top).to(device)

    optimizer = optim.Adamax(model.parameters(), lr=p.lr, weight_decay=p.weight_decay)    

    data = np.load(args.data_path).reshape(-1, p.imShape[0],64,64)[:10000][:,:,::4,::4]
    data = (data - np.mean(data)) / (np.std(data))
    features = torch.tensor(data, dtype=torch.float32)  # Convert features to PyTorch tensor
    dataset = TensorDataset(features)  # Only features are included
    loader = DataLoader(dataset, batch_size=p.batch_size, shuffle=False, pin_memory=True)
    plt.figure()
    print(data[0].shape)
    plt.imshow(data[0].squeeze())
    plt.colorbar()
    plt.show()

    lowest = 1e7
    patience = 0
    model.train()
    epochs = 50

    for ep in range(epochs):
        ep_loss = []
        for idx, (x,) in enumerate(loader):
            print(f"Epoch: {ep} Progress: {round((idx * 100) / (len(loader)), 4)}% Likelihood:      {lowest} Patience:      {patience}" , end="\r")
            x = x.to(device)
            optimizer.zero_grad()

            x += p.noise_level * torch.randn_like(x).to(device)
            z, likelihood = model(x)

            loss = torch.mean(-likelihood)
            loss.backward()
            optimizer.step()
            loss_ = loss.detach().cpu().numpy()
            ep_loss.append(loss_)
            if idx == 2000:
                torch.save(model.state_dict(), f'saves/{fName}-kappa.pt')
                print(ep_loss)
                plt.figure()
                plt.plot(ep_loss)
                plt.savefig('ep_loss.png')
                plt.close()
                exit()

        avg_loss = round(float(np.mean(ep_loss)), 2)
        #if lowest > avg_loss:    
        lowest = avg_loss
        torch.save(model.state_dict(), f'saves/{fName}-kappa.pt')

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default="data/test_kappa_[3]kev_0.npy", help='train level')
args = parser.parse_args()
main(args)