import torch
import numpy as np
import matplotlib.pyplot as plt
from model import InvariantFlowModel
from importlib.machinery import SourceFileLoader
import argparse

def encode(args, model, device, num_samples=8, image_shape=(1, 64, 64)):

    data = np.load(args.data_path).reshape(-1, image_shape[0], image_shape[1], image_shape[2])
    data = (data - np.mean(data)) / (np.std(data))
    data = torch.tensor(data, dtype=torch.float32) 

    x = data[0:num_samples].to(device)
    # Add optional noise
    x += p.noise_level * torch.randn_like(x).to(device)

     # Reverse transform to get data samples
    z_samples, _ = model(x=x, reverse=False)
    z_samples = z_samples.detach().cpu().numpy()

    fig, axs = plt.subplots(2, num_samples, figsize=(15, 4))

    # Plot x samples in the first row
    for i in range(num_samples):
        im = axs[0, i].imshow(x[i].cpu().numpy().squeeze())
        axs[0, i].axis('off')  # Turn off axis

    # Plot z samples in the second row
    for i in range(num_samples):
        im = axs[1, i].imshow(z_samples[i].squeeze())
        axs[1, i].axis('off')  # Turn off axis

    fig.colorbar(im, ax=axs, orientation='horizontal', fraction=0.02, pad=0.04)
    plt.show()

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='data path')
    parser.add_argument('--model_path', type=str, help='model path')
    args = parser.parse_args()

    p = SourceFileLoader('cf', 'config.py').load_module()
    model = InvariantFlowModel(image_shape=p.imShape, n_layers=p.n_layers, learn_top=p.y_learn_top).to(device)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad), ' Parameters')
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
    model.eval() 
    encode(args, model, device, image_shape=p.imShape)