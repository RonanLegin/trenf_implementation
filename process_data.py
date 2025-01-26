import argparse
import numpy as np
import torch
import ptwt, pywt # pip install ptwt



def load_data(file_path):
    return torch.from_numpy(np.load(file_path))

def save_data(data, file_path):
    np.save(file_path, data)

def main():
    parser = argparse.ArgumentParser(description="Apply wavelet transform to data.")
    parser.add_argument("--input_file", type=str, help="Path to the input file")
    parser.add_argument("--level", type=int, help="Level of wavelet decomposition")
    args = parser.parse_args()

    data = load_data(args.input_file)
    coefficients = ptwt.wavedec2(data, pywt.Wavelet("haar"), level=args.level, mode="zero")
    x = coefficients[0].detach().cpu().numpy()  # Low resolution image

    output_file = f"{args.input_file.rsplit('.', 1)[0]}_level{args.level}.npy"
    save_data(x, output_file)

if __name__ == "__main__":
    main()