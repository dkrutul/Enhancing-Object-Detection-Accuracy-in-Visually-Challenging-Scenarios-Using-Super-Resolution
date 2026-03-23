import glob
import os
import sys

import cv2
import numpy as np
import torch
from core.models import RLFN_S
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm


def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def process_image(model, img, device):
    input_tensor = to_tensor(img).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor.to(device))
    output_image = output.cpu().squeeze(0).permute(1, 2, 0).numpy() * 255
    output_image = output_image.clip(0, 255).astype(np.uint8)
    return output_image

def evaluate_images(model, data_path, device, upscale):
    psnr_list, mse_list, ssim_list = [], [], []

    for full_path in tqdm(data_path):
        original_image = load_image(full_path)
        img_size = (int(original_image.shape[1] * (1/upscale)), int(original_image.shape[0] * (1/upscale)))
        downsize_image = cv2.resize(original_image, dsize=img_size, interpolation=cv2.INTER_CUBIC)
        super_res_image = process_image(model, downsize_image, device)

        psnr_value = peak_signal_noise_ratio(original_image, super_res_image)
        mse_value = mean_squared_error(original_image, super_res_image)
        
        psnr_list.append(psnr_value)
        mse_list.append(mse_value)
        
    return np.mean(psnr_list), np.mean(mse_list), []

if __name__ == "__main__":

    # Base directory
    base_dir = './soccernet/tracking/test/'

    # Pattern to match all .jpg files in the specified subdirectories
    pattern = os.path.join(base_dir, '*/img1/*.jpg')

    # Find all files matching the pattern
    jpg_files = glob.glob(pattern)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    with open('results/sr_quality_lr_modified_soccernet.txt', 'w') as file:
        original_stdout = sys.stdout  
        # Redirect stdout to the file
        sys.stdout = file

        print("Trained on SoccerNet")
        for upscale in [3, 4, 6, 2]:
            model = RLFN_S(in_channels=3, out_channels=3, upscale=upscale)
            LR = 0.01
            model.load_state_dict(torch.load(f"model/save_x{upscale}_15000_32_{LR}_sr_soccernet_train_LRdim128.pth"))
            model = model.to(device)

            psnr, mse, ssim = evaluate_images(model, jpg_files, device, upscale)
            print(upscale, psnr, mse)
            print(f"Mean PSNR/MSE for x{upscale}: {psnr}, {mse}")
