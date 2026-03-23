import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

categories_label2id = {
    "person": 0,
    "sports ball": 1
}

categories_id2label = {
    0: "person",
    1: "sports ball"
}

def img_to_torch(img):
    """Convert PIL or ndarray (H,W,C) to Tensor (1,C,H,W)"""
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img.copy()).float()
    if img.max() > 1.0: img /= 255.0
    return img.permute(2, 0, 1).unsqueeze(0)

def torch_to_img(tensor):
    """Convert Tensor (1,C,H,W) to ndarray (H,W,C)"""
    img = tensor.detach().cpu().squeeze().clamp(0, 1).numpy()
    return np.transpose(img, (1, 2, 0))

def read_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img

def read_rgb_image(path):
    """Utility function that loads an image as an RGB numpy aray."""
    return np.asarray(Image.open(path))

def uint2tensor4(img):
    img = img.numpy()
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0)

def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())

def calculate_iou(box1, box2):
    """IoU for bbox format [xmin, ymin, width, height]"""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[0] + box1[2], box2[0] + box2[2])
    y_bottom = min(box1[1] + box1[3], box2[1] + box2[3])
    
    if x_right < x_left or y_bottom < y_top: return 0.0
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1, area2 = box1[2]*box1[3], box2[2]*box2[3]
    return intersection / float(area1 + area2 - intersection)

def preprocess_data(sample):
    img_path = sample['img_path']
    image = F.to_tensor(Image.open(img_path).convert("RGB"))
    return image, sample['ground_truth'], img_path

def format_ground_truth(ground_truth, img_id, img_path):
    """Function to format the ground truths for COCO evaluation"""
    annotations = []
    for gt in ground_truth:
        annotations.append({
            "image_id": img_id,
            "file_name": img_path,
            "category_id": categories_label2id[gt['label']],
            "bbox": gt['bbox'], # if not format_bbox else convert_bbox_format(gt['bbox']),
            "score": 1.0,
            "iscrowd": 0,
            "area": gt['bbox'][-1]*gt['bbox'][-2]
        })
    return annotations

def convert_to_coco(ground_truths, h=1920, w=1080, format_bbox=False):
    image_ids = set()
    category_names = set()

    for i, item in enumerate(ground_truths):
        item["id"] = i
        image_ids.add(item["image_id"])

    # Create images and categories data
    images = [{"id": image_id, "height": h, "weight": w, "area": h*w} for image_id in image_ids]
    categories = [{"id": i, "name": name} for name, i in categories_label2id.items()] #i, name in enumerate(category_names)]

    # Update annotations with category ids
    for item in ground_truths:
        item["bbox"] = item['bbox'] if not format_bbox else tv.ops.box_convert(torch.Tensor(item['bbox']), in_fmt='xyxy', out_fmt='xywh').tolist()

    # Combine everything into COCO format
    coco_format = {
        "images": images,
        "annotations": ground_truths,
        "categories": categories
    }

    return coco_format

def bbox_iou(box1, box2):
    """Compute the IoU of two bounding boxes"""
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[0] + box1[2], box2[0] + box2[2])
    y_bottom = min(box1[1] + box1[3], box2[1] + box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate the area of intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate the area of both bounding boxes
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]

    # Compute the IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def compute_mean_iou(coco_gt, coco_dt, iou_threshold=0.5):

    ious = []
    for img_id in coco_gt.getImgIds():
        # Retrieve the annotations for the current image
        gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
        dt_anns = coco_dt.loadAnns(coco_dt.getAnnIds(imgIds=img_id))

        for dt in dt_anns:
            best_iou = 0
            for gt in gt_anns:
                iou = bbox_iou(dt['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
            
            # Add the best IoU for this detection if it meets the threshold
            if best_iou >= iou_threshold:
                ious.append(best_iou)

    mean_iou = np.mean(ious) if ious else 0
    return mean_iou

def preprocess_data_downscale(sample, downscale=1):
    img_path = sample['img_path']
    image = Image.open(img_path).convert("RGB")
    width, height = image.size 
    newsize = (int(width/downscale), int(height/downscale))
    image = image.resize(newsize)
    image = F.to_tensor(image)
    
    for gt in sample['ground_truth']:
        gt['bbox'] = [int(x / downscale) for x in gt['bbox']]

    return image, sample['ground_truth'], img_path

def preprocess_data_imsize(sample, im_size):
    img_path = sample['img_path']
    image = Image.open(img_path).convert("RGB")
    width, height = image.size 
    image = image.resize(im_size)
    new_w, new_h = image.size
    scale_w, scale_h = int(width/new_w), int(height/new_h)
    image = F.to_tensor(image)
    
    for gt in sample['ground_truth']:
        # x,y,w,h
        gt['bbox'] = [
            int(gt['bbox'][0] / scale_w), 
            int(gt['bbox'][1] / scale_h), 
            int(gt['bbox'][2] / scale_w), 
            int(gt['bbox'][3] / scale_h), 
        ]

    return image, sample['ground_truth'], img_path
