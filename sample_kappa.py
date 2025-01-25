import torch
import numpy as np
import matplotlib.pyplot as plt
from model import InvariantFlowModel  # Assuming your model's class is named this and imported properly
from importlib.machinery import SourceFileLoader

def sample_kappa(model, device, num_samples=8, image_shape=(1, 64, 64)):
    #model.eval()  # Set the model to evaluation mode
    
    # Sample from the Gaussian prior: model.prior might need to be adjusted based on your actual model implementation
    mean = torch.zeros(num_samples, *image_shape).to(device)  # mean is zero
    logs = torch.zeros(num_samples, *image_shape).to(device)  # log standard deviation is zero (standard deviation is one)
    z = mean + torch.exp(logs) * torch.randn_like(mean)  # z = mean + std * N(0, I)
    print(z.shape)
    # Reverse transform to get data samples
    x_samples, _ = model(z=z, reverse=True)

    # Convert samples to numpy and reshape for plotting
    x_samples = x_samples.detach().cpu().numpy()
    print(x_samples)
    # Plotting
    fig, axs = plt.subplots(1, num_samples, figsize=(15, 2))  # Adjust size as needed
    for i in range(num_samples):
        im = axs[i].imshow(x_samples[i].squeeze())  # Assuming grayscale images
        axs[i].axis('off')  # Turn off axis
    fig.colorbar(im, ax=axs, orientation='horizontal', fraction=0.02, pad=0.04)  # Add colorbar
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p = SourceFileLoader('cf', 'config_kappa.py').load_module()
    model = InvariantFlowModel(image_shape=p.imShape, n_layers=p.n_layers, learn_top=p.y_learn_top).to(device)

    model.load_state_dict(torch.load('saves/20250125_12_00-kappa.pt'))  # Update the path to your model
    
    sample_kappa(model, device, image_shape=p.imShape)