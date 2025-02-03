import os
import torch
import random
import numpy as np
from importlib.machinery import SourceFileLoader
from torch.utils.data import TensorDataset, DataLoader

import torch.optim as optim
from trenf import InvariantFlowModel

import argparse
import time


seed = 786
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    fName = time.strftime("%Y%m%d_%H_%M")
    if not os.path.exists("saves/"):
        os.makedirs("saves/")

    p = SourceFileLoader('cf', 'config.py').load_module()
    model = InvariantFlowModel(image_shape=p.imShape, n_layers=p.n_layers, n_kernel_knots=p.n_kernel_knots, n_nonlinearity_knots=p.n_nonlinearity_knots, learn_top=p.y_learn_top).to(device)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad), ' Parameters')
    model.train()

    optimizer = optim.Adamax(model.parameters(), lr=p.lr, weight_decay=p.weight_decay)    

    data = np.load(args.data_path).reshape(-1, p.imShape[0], p.imShape[1], p.imShape[2])
    data = (data - np.mean(data)) / (np.std(data))
    features = torch.tensor(data, dtype=torch.float32)  # Convert features to PyTorch tensor
    dataset = TensorDataset(features)  # Only features are included
    loader = DataLoader(dataset, batch_size=p.batch_size, shuffle=False, pin_memory=True)

    lowest = 1e7
    model.train()
    epochs = 2000

    for ep in range(epochs):
        ep_loss = []
        for idx, (x,) in enumerate(loader):
            print(f"Epoch: {ep} Progress: {round((idx * 100) / (len(loader)), 4)}% Likelihood:      {lowest}" , end="\r")
            x = x.to(device)
            optimizer.zero_grad()

            # Optional noise to improve stability
            x += p.noise_level * torch.randn_like(x).to(device)

            z, likelihood = model(x)

            loss = torch.mean(-likelihood)
            loss.backward()
            optimizer.step()
            loss_ = loss.detach().cpu().numpy()
            ep_loss.append(loss_)

        avg_loss = round(float(np.mean(ep_loss)), 2)

        if lowest > avg_loss:    
            lowest = avg_loss
            torch.save(model.state_dict(), f'saves/{fName}-model.pt')

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='train data path')
args = parser.parse_args()
main(args)