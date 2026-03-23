import json
import sys

import torch
from core.dataset import (SoccerNetModified, SoccerNetModifiedSR,
                          load_tracking_data)
from core.models import RLFN_S
from core.utils import compute_mean_iou, convert_to_coco
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_V2_Weights,
                                          fasterrcnn_resnet50_fpn_v2)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

batch_size = 8
categories_label2id = {
    "person": 0,
    "sports ball": 1
}

categories_id2label = {
    0: "person",
    1: "sports ball"
}

# test
print("Loading test")
tracking_data = load_tracking_data("soccernet/tracking/test")
transform = None
print("Datasets loaded")



with open(f'results/cocoEval_output_imsize_sr.txt', 'a') as file:
    original_stdout = sys.stdout  
    # Redirect stdout to the file
    sys.stdout = file
    train_name = "SoccerNet"
    print("\n\n\n\n\n\n")
    print(f"SR Train: {train_name}")
    for upscale in [
        2, 3, 4, 6
    ]:
        print("")
        print("----------------------------------------")
        print("Upscale: ", upscale)
        # Load the model
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
    
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        num_classes = len(categories_label2id) + 1
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=num_classes)


        model.load_state_dict(torch.load('model/faster_rcnn_5_1920_1080.pth'))


        model = model.to('cuda')

        # Load SR model
        sr_model = RLFN_S(in_channels=3, out_channels=3, upscale=upscale)

        sr_model.load_state_dict(torch.load(f"model/save_x{upscale}_15000_32_0.01_sr_soccernet_train_LRdim128.pth"))

        model.eval()
        new_im_size = (1280, 720)
        uscaled_im_size = (new_im_size[0] * upscale, new_im_size[1] * upscale)
        print(f"New image size: {new_im_size}")
        print(f"Upscaled image size: {uscaled_im_size}")
        test_dataset_raw = SoccerNetModified(tracking_data, new_im_size=new_im_size)
        test_dataset_upscaled = SoccerNetModifiedSR(tracking_data, new_im_size=new_im_size, sr_model=sr_model, upscale=upscale)
        print("Prediction")
        # DataLoader
        batch_size_2 = 4
        test_data_loader_raw = DataLoader(test_dataset_raw, batch_size=batch_size_2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
        test_data_loader_upscaled = DataLoader(test_dataset_upscaled, batch_size=batch_size_2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))        
            
        # Predict and format model outputs
        model_predictions = []
        ground_truths = []
        
        categories = [{"id": id, "name": name} for name, id in categories_label2id.items()]
        coco_output_gt = {
            "images": [],
            "annotations": [],
            "categories": categories
        }
        for name, loader in zip(["test_data_loader_upscaled", "test_data_loader_raw"], [test_data_loader_upscaled, test_data_loader_raw]):
            print(name)
            annotation_id = 1
            annotation_gt_id = 1 
            coco_output_pred = []
            for idx, (images, targets) in tqdm(enumerate(loader)):
                images = list(image.to('cuda') for image in images)
                outputs = model(images)

                for i, output in enumerate(targets):
                    image_id = idx * batch_size_2 + i
                    for box, label in zip(output['boxes'], output['labels']):
                        x_min, y_min, x_max, y_max = box.tolist()
                        width, height = x_max - x_min, y_max - y_min
                        coco_output_gt["annotations"].append({
                            "id": annotation_gt_id,
                            "image_id": image_id,
                            "category_id": label.item(),
                            "bbox": [x_min, y_min, width, height],
                            "area": width * height,
                            "iscrowd": 0,
                        })
                        annotation_gt_id += 1

                for i, output in enumerate(outputs):
                    # Assuming that the test dataset has an attribute 'get_image_id' to get the image ID
                    image_id = idx * batch_size_2 + i

                    # Add image info
                    coco_output_gt["images"].append({
                        "id": image_id,
                        "file_name": f"image_{image_id}.jpg",
                        "height": new_im_size[1],
                        "width": new_im_size[0]
                    })

                    # Add annotations
                    for box, label, score in zip(output['boxes'], output['labels'], output['scores']):
                        x_min, y_min, x_max, y_max = box.tolist()
                        width, height = x_max - x_min, y_max - y_min

                        coco_output_pred.append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": label.item(),
                            "bbox": [x_min, y_min, width, height],
                            "area": width * height,
                            "iscrowd": 0,
                            "score": score.item()
                        })
                        annotation_id += 1

            # Save predictions and ground truths as JSON
            coco_ground_truths = convert_to_coco(ground_truths)
            with open('ground_truths.json', 'w') as f:
                json.dump(coco_output_gt, f)

            coco_model_predictions = convert_to_coco(model_predictions, format_bbox=True)
            with open(f'results/predictions_{train_name}_{new_im_size[0]}_{new_im_size[1]}_{upscale}.json', 'w') as f:
                json.dump(coco_output_pred, f)
            print("len(coco_output_pred)", len(coco_output_pred))
            if len(coco_output_pred) > 0:
                # Load and evaluate using COCO API
                cocoGt = COCO("ground_truths.json")
                cocoDt = cocoGt.loadRes(f'results/predictions_{train_name}_{new_im_size[0]}_{new_im_size[1]}_{upscale}.json')

                cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')

                cocoEval.evaluate()
                cocoEval.accumulate()
                print("new_im_size:", new_im_size)
                print(cocoEval.summarize())
                for iou_threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                    mean_iou = compute_mean_iou(cocoGt, cocoDt, iou_threshold)
                    print(f"Mean IoU (for IoU >= {iou_threshold}): {mean_iou}")
                print("-----------------\n")
            else:
                print("No predictions")
        
    sys.stdout = original_stdout
