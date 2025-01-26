import torch
import numpy as np
import matplotlib.pyplot as plt
from model import InvariantFlowModel
from importlib.machinery import SourceFileLoader
import argparse 

def sample(model, num_samples=8):
    model.eval() 
    
    mean, logs = model.prior(num_samples)
    z = mean + torch.exp(logs) * torch.randn_like(mean)  # z = mean + std * N(0, I)

    # Reverse transform to get data samples
    x_samples, _ = model(z=z, reverse=True)
    x_samples = x_samples.detach().cpu().numpy()

    # Plotting
    fig, axs = plt.subplots(1, num_samples, figsize=(15, 2))
    for i in range(num_samples):
        im = axs[i].imshow(x_samples[i].squeeze())
        axs[i].axis('off')  # Turn off axis
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
    sample(model, device, image_shape=p.imShape)