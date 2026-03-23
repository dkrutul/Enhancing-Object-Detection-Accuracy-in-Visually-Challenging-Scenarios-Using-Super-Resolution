import os

import numpy as np
import pandas as pd
import torch
import torchvision as tv
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

categories_label2id = {
    "person": 0,
    "sports ball": 1
}

categories_id2label = {
    0: "person",
    1: "sports ball"
}

def extract_ball_ids(file_path):
    ball_ids = []

    with open(file_path, 'r') as file:
        for line in file:
            if 'ball;' in line:
                # Extracting the part after 'ball;'
                parts = line.strip().split('= ball;')
                if len(parts) > 1:
                    ball_id = parts[0]
                    ball_id = int(ball_id.replace("trackletID_", ""))
                    ball_ids.append(ball_id)

    return ball_ids


def load_tracking_data(base_path='tracking/train'):
    data = []
    for subdir in os.listdir(base_path):
        subdir_path = os.path.join(base_path, subdir)
        gt_path = os.path.join(subdir_path, 'gt', 'gt.txt')

        if os.path.exists(gt_path):
            gt_data = pd.read_csv(gt_path, header=None, sep=',')
            gt_data.columns = ['frame_id', 'track_id', 'top_x', 'top_y', 'width', 'height', 'confidence', 'unused1', 'unused2', 'unused3']

            # Add ball vs person labels
            ball_ids = extract_ball_ids(os.path.join(subdir_path, 'gameinfo.ini'))
            gt_data['track_id'] = ["sports ball" if elem in ball_ids else "person" for elem in gt_data['track_id'].values]

            gt_data['image_path'] = gt_data['frame_id'].apply(lambda x: os.path.join(subdir_path, 'img1', f'{x:06d}.jpg'))

            grouped_data = gt_data.groupby(['image_path', 'frame_id']).apply(lambda x: x[['track_id', 'top_x', 'top_y', 'width', 'height']].to_dict('records')).reset_index()
            grouped_data = grouped_data.rename(columns={0: 'ground_truth'})

            data.extend(grouped_data.to_dict('records'))

    return data

class SoccerNet(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.data[idx]['image_path']

        frame_id = self.data[idx]['frame_id']
        img_path = self.data[idx]['image_path']

        sample = {
            'frame_id':frame_id, 
            'img_path': img_path,
            'height': 1080,
            'width': 1920,
            'ground_truth': [
                {
                    'label': elem['track_id'],
                    'category_id': categories_label2id[elem['track_id']],
                    'bbox': [elem['top_x'], elem['top_y'], elem['width'], elem['height']]
                } for elem in self.data[idx]['ground_truth']
            ]}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample


class SoccerNetModified(SoccerNet):
    def __init__(self, data, transform=None, new_im_size=(1920, 1080)):
        self.data = data
        self.transform = transform
        self.new_im_size = new_im_size
        self.scale_x = 1920/new_im_size[0] 
        self.scale_y = 1080/new_im_size[1] 
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.data[idx]['image_path']
        image = Image.open(img_path).convert('RGB')
        image = image.resize(self.new_im_size)
        if self.transform:
            image = self.transform(image)
        else:
            image = tv.transforms.functional.to_tensor(image)
        frame_id = self.data[idx]['frame_id']
        gt = [
            {
                'label': elem['track_id'],
                'category_id': categories_label2id[elem['track_id']],
                'bbox': [
                    elem['top_x']/self.scale_x,
                    elem['top_y']/self.scale_y,
                    (elem['width']/self.scale_x) + (elem['top_x']/self.scale_x),
                    (elem['height']/self.scale_y) + (elem['top_y']/self.scale_y)
                ]
            } for elem in self.data[idx]['ground_truth'] if (
                (elem['top_x']/self.scale_x != (elem['width']/self.scale_x) + (elem['top_x']/self.scale_x)) and  (elem['top_y']/self.scale_y != (elem['height']/self.scale_y) + (elem['top_y']/self.scale_y)))
        ]

        targets = [{
            'boxes': torch.tensor([elem['bbox'] for elem in gt], dtype=torch.float32),
            'labels': torch.tensor([elem['category_id'] for elem in gt], dtype=torch.int64)
        }]
        

        return image, targets[0]


def img_to_torch(img):
    img_np = np.array(img, dtype=np.float32)

    # Convert NumPy array to PyTorch tensor
    img_tensor = torch.from_numpy(img_np)

    # Rearrange the tensor dimensions
    # From [H, W, C] to [1, C, H, W] for batch size of 1
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

    return img_tensor

def torch_to_img(torch_image):
    return torch_image.swapaxes(1, 2).swapaxes(0, 2).detach().numpy()

class SoccerNetModifiedSR(SoccerNet):
    def __init__(self, data, sr_model, upscale,
                 transform=None, new_im_size=(1920, 1080)):
        """
        Args:
            data (list): Lista słowników z danymi wczytanymi z plików gt.txt.
            transform (callable, optional): Opcjonalna transformacja do zastosowania na obrazach.
        """
        self.data = data
        self.transform = transform
        self.new_im_size = new_im_size
        self.scale_x = (1920/new_im_size[0]) * upscale #int(1920/new_im_size[0])
        self.scale_y = (1080/new_im_size[1]) * upscale #int(1080/new_im_size[1])
        self.sr_model = sr_model
        self.upscale = upscale
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.data[idx]['image_path']
        image = Image.open(img_path).convert('RGB')
        image = image.resize(self.new_im_size)
        torch_img = img_to_torch(image)
        torch_sr_img = self.sr_model(torch_img)
        image = torch_to_img(torch_sr_img[0])
        if self.transform:
            image = self.transform(image)
        else:
            image = tv.transforms.functional.to_tensor(image)
        frame_id = self.data[idx]['frame_id']
        # rescale bbox
        gt = [
            {
                'label': elem['track_id'],
                'category_id': categories_label2id[elem['track_id']],
                'bbox': [
                    elem['top_x']/self.scale_x,
                    elem['top_y']/self.scale_y,
                    (elem['width']/self.scale_x) + (elem['top_x']/self.scale_x),
                    (elem['height']/self.scale_y) + (elem['top_y']/self.scale_y)
                ]
            } for elem in self.data[idx]['ground_truth'] if (
                (elem['top_x']/self.scale_x != (elem['width']/self.scale_x) + (elem['top_x']/self.scale_x)) and  (elem['top_y']/self.scale_y != (elem['height']/self.scale_y) + (elem['top_y']/self.scale_y)))
        ]

        targets = [{
            'boxes': torch.tensor([elem['bbox'] for elem in gt], dtype=torch.float32),
            'labels': torch.tensor([elem['category_id'] for elem in gt], dtype=torch.int64)
        }]
        

        return image, targets[0]
