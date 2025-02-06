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
from torch.utils.data import Dataset


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
    model = InvariantFlowModel(
        image_shape=p.imShape,
        n_layers=p.n_layers,
        n_kernel_knots=p.n_kernel_knots,
        n_nonlinearity_knots=p.n_nonlinearity_knots,
        learn_top=p.y_learn_top,
        conditional=True,
        num_conditions=len(args.data_paths),
    ).to(device)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad), ' Parameters')
    model.train()

    optimizer = optim.Adamax(model.parameters(), lr=p.lr, weight_decay=p.weight_decay)    

    data_list = []
    for i, data_path in enumerate(args.data_paths):
        arr = np.load(data_path).reshape(-1, p.imShape[0], p.imShape[1], p.imShape[2])
        arr = (arr - np.mean(arr)) / np.std(arr)
        data_tensor = torch.tensor(arr, dtype=torch.float32)
        data_list.append((data_tensor, i))

    # class RandomDataset(Dataset):
    #     def __init__(self, datasets):
    #         self.datasets = datasets
    #     def __len__(self):
    #         return 10**6  # Arbitrary large number
    #     def __getitem__(self, _):
    #         ds_idx = random.randrange(len(self.datasets))
    #         x_tensor, cond = self.datasets[ds_idx]
    #         idx_in_ds = random.randrange(x_tensor.shape[0])
    #         return x_tensor[idx_in_ds], cond

    # dataset = RandomDataset(data_list)
    # loader = DataLoader(dataset, batch_size=p.batch_size, shuffle=False, pin_memory=True)

    class RandomBatchDataset(Dataset):
        def __init__(self, datasets, batch_size):
            self.datasets = datasets
            self.batch_size = batch_size

        def __len__(self):
            # This defines how many 'batches' we can draw (somewhat arbitrary)
            return 10**2

        def __getitem__(self, _):
            # Select one dataset randomly
            ds_idx = random.randrange(len(self.datasets))
            x_tensor, cond = self.datasets[ds_idx]

            # Select a random batch from the selected dataset
            idxs = torch.randint(0, x_tensor.size(0), (self.batch_size,))
            batch = x_tensor[idxs]

            return batch, torch.tensor(cond, dtype=torch.long)
        
    dataset = RandomBatchDataset(data_list, p.batch_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)  # Each item is already a batch

    # Simple custom collate function to avoid unnecessary batching
    def custom_collate(batch):
        # 'batch' is a list of tuples (data, condition)
        # Since each item in 'batch' is already a full batch, we just take the first one
        data, condition = batch[0]
        return data, condition

    loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, collate_fn=custom_collate)

    lowest = 1e7
    model.train()
    epochs = 2000

    for ep in range(epochs):
        ep_loss = []
        for idx, (x, condition) in enumerate(loader):
            print(f"Epoch: {ep} Progress: {round((idx * 100) / (len(loader)), 4)}% Likelihood:      {lowest}" , end="\r")
            x = x.to(device)
            optimizer.zero_grad()

            print(condition)
            # Optional noise to improve stability
            x += p.noise_level * torch.randn_like(x).to(device)

            z, likelihood = model(x, condition=condition)

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
parser.add_argument('--data_paths', type=str, nargs='+', help='List of train data paths')
args = parser.parse_args()
main(args)