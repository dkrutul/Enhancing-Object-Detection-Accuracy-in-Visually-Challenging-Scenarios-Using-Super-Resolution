import os
import json
import torch
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import yaml
from collections import defaultdict
import glob

def setup_device():
    """Configures GPU/CPU."""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("⚠️ Using CPU")
    return device

def load_yolov9_model(model_path, device='cpu'):
    """Loads a trained YOLOv9 model."""
    try:
        print("🔧 Loading YOLOv9 model...")
        print(f"📁 Path: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return None
        
        # Load model - YOLOv9 uses torch.hub or ultralytics
        try:
            # Try ultralytics first
            from ultralytics import YOLO
            model = YOLO(model_path)
            print("✅ Loaded via ultralytics")
        except ImportError:
            # Fallback to torch.hub
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, device=device)
            print("✅ Loaded via torch.hub")
        
        print(f"📊 Model loaded on {device}")
        return model
    
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_class_names():
    """Loads class names from dataset.yaml."""
    yaml_path = "yolo_dataset/data.yaml"
    
    if os.path.exists(yaml_path):
        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            class_names = data.get('names', {})
            print(f"✅ Loaded class names: {class_names}")
            return class_names
        except Exception as e:
            print(f"⚠️ Failed to load data.yaml: {e}")
    
    # Fallback - ExDark classes
    class_names = {
        0: 'Bicycle', 1: 'Boat', 2: 'Bottle', 3: 'Bus', 4: 'Car', 5: 'Cat',
        6: 'Chair', 7: 'Cup', 8: 'Dog', 9: 'Motorbike', 10: 'People', 11: 'Table'
    }
    print(f"⚠️ Using default class names: {class_names}")
    return class_names

def load_test_data():
    """Loads test data."""
    test_images_dir = "hat_dataset/test/SR_4x_pretrained"   # yolo_dataset/images/test
    test_labels_dir = "yolo_dataset/labels/test"
    
    if not os.path.exists(test_images_dir):
        print(f"❌ Test images folder not found: {test_images_dir}")
        return [], []
    
    if not os.path.exists(test_labels_dir):
        print(f"❌ Test labels folder not found: {test_labels_dir}")
        return [], []
    
    # Find images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(test_images_dir, ext)))
    
    # Find matching labels
    test_data = []
    
    for image_path in image_paths:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(test_labels_dir, f"{image_name}.txt")
        
        if os.path.exists(label_path):
            test_data.append((image_path, label_path))
        else:
            print(f"⚠️ Missing label for: {image_name}")
    
    print(f"✅ Found {len(test_data)} image-label pairs")
    return test_data

def parse_yolo_label(label_path, img_width, img_height):
    """Parses a YOLO label file into bounding-box coordinates."""
    bboxes = []
    
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Convert from YOLO format to pixel coordinates
                x_min = (x_center - width/2) * img_width
                y_min = (y_center - height/2) * img_height
                x_max = (x_center + width/2) * img_width
                y_max = (y_center + height/2) * img_height
                
                bboxes.append({
                    'class_id': class_id,
                    'bbox': [x_min, y_min, x_max, y_max],
                    'area': width * height * img_width * img_height
                })
    
    except Exception as e:
        print(f"❌ Label parsing error {label_path}: {e}")
    
    return bboxes

def calculate_iou(box1, box2):
    """Calculates IoU between two bounding boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Union area
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    
    if union_area <= 0:
        return 0.0
    
    return inter_area / union_area

def match_predictions_to_ground_truth(predictions, ground_truth, iou_threshold=0.5):
    """Matches predictions to ground truth."""
    matches = []
    used_gt = set()
    
    # Sort predictions by confidence (descending)
    predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
    
    for pred in predictions:
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in used_gt:
                continue
            
            # Check class consistency
            if pred['class_id'] != gt['class_id']:
                continue
            
            iou = calculate_iou(pred['bbox'], gt['bbox'])
            
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_gt_idx >= 0:
            matches.append({
                'prediction': pred,
                'ground_truth': ground_truth[best_gt_idx],
                'iou': best_iou,
                'true_positive': True
            })
            used_gt.add(best_gt_idx)
        else:
            matches.append({
                'prediction': pred,
                'ground_truth': None,
                'iou': 0.0,
                'true_positive': False
            })
    
    return matches, used_gt

def calculate_precision_recall(matches, total_ground_truth):
    """Calculates precision and recall."""
    true_positives = sum(1 for match in matches if match['true_positive'])
    false_positives = sum(1 for match in matches if not match['true_positive'])
    false_negatives = total_ground_truth - true_positives
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    return precision, recall

def calculate_ap(precisions, recalls):
    """Calculates Average Precision (AP)."""
    # Add points (0,0) and (1,0)
    recalls = [0] + recalls + [1]
    precisions = [0] + precisions + [0]
    
    # Interpolation
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # Compute AP
    ap = 0
    for i in range(1, len(recalls)):
        if recalls[i] != recalls[i - 1]:
            ap += (recalls[i] - recalls[i - 1]) * precisions[i]
    
    return ap

def run_model_inference(model, image_path):
    """Runs model inference on an image."""
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Cannot load image: {image_path}")
            return []
        
        # Run prediction
        results = model(image_path)
        
        predictions = []
        
        # Parse outputs (different formats for different YOLO versions)
        if hasattr(results, 'pandas'):
            # YOLOv5 format
            df = results.pandas().xyxy[0]
            for _, row in df.iterrows():
                predictions.append({
                    'class_id': int(row['class']),
                    'confidence': float(row['confidence']),
                    'bbox': [float(row['xmin']), float(row['ymin']),
                            float(row['xmax']), float(row['ymax'])]
                })
        elif hasattr(results, 'boxes'):
            # Ultralytics YOLO format
            if results.boxes is not None:
                boxes = results.boxes
                for i in range(len(boxes)):
                    predictions.append({
                        'class_id': int(boxes.cls[i]),
                        'confidence': float(boxes.conf[i]),
                        'bbox': boxes.xyxy[i].tolist()
                    })
        else:
            # Fallback - try direct access
            try:
                for detection in results:
                    if hasattr(detection, 'boxes'):
                        boxes = detection.boxes
                        for i in range(len(boxes)):
                            predictions.append({
                                'class_id': int(boxes.cls[i]),
                                'confidence': float(boxes.conf[i]),
                                'bbox': boxes.xyxy[i].tolist()
                            })
            except:
                print(f"⚠️ Unknown result format for {image_path}")
        
        return predictions
    
    except Exception as e:
        print(f"❌ Inference error for {image_path}: {e}")
        return []

def evaluate_model(model, test_data, class_names, confidence_threshold=0.25):
    """Evaluates the model on test data."""
    print("\n🎯 MODEL EVALUATION")
    print("=" * 50)
    
    # IoU thresholds for different metrics
    iou_thresholds = {
        'mAP@0.50:0.95': np.arange(0.5, 1.0, 0.05),
        'mAP@0.50': [0.5],
        'τIoU=0.5': [0.5],
        'τIoU=0.7': [0.7],
        'τIoU=0.9': [0.9]
    }
    
    all_predictions = []
    all_ground_truth = []
    all_ious = []
    
    results_per_class = defaultdict(lambda: {
        'true_positives': defaultdict(int),
        'false_positives': defaultdict(int),
        'false_negatives': defaultdict(int),
        'total_ground_truth': 0,
        'ious': []
    })
    
    print(f"🔍 Processing {len(test_data)} images...")
    
    for idx, (image_path, label_path) in enumerate(test_data):
        print(f"📷 {idx+1}/{len(test_data)}: {os.path.basename(image_path)}")
        
        # Load image to get dimensions
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        img_height, img_width = image.shape[:2]
        
        # Load ground truth
        ground_truth = parse_yolo_label(label_path, img_width, img_height)
        
        # Run prediction
        predictions = run_model_inference(model, image_path)
        
        # Filter predictions by confidence
        predictions = [p for p in predictions if p['confidence'] >= confidence_threshold]
        
        all_predictions.extend(predictions)
        all_ground_truth.extend(ground_truth)
        
        # Evaluate for each IoU threshold
        for metric_name, thresholds in iou_thresholds.items():
            for threshold in thresholds:
                matches, used_gt = match_predictions_to_ground_truth(
                    predictions, ground_truth, threshold
                )
                
                # Collect IoU for mean IoU
                for match in matches:
                    if match['true_positive']:
                        all_ious.append(match['iou'])
                        results_per_class[match['prediction']['class_id']]['ious'].append(match['iou'])
                
                # Count class-wise metrics
                for match in matches:
                    class_id = match['prediction']['class_id']
                    if match['true_positive']:
                        results_per_class[class_id]['true_positives'][threshold] += 1
                    else:
                        results_per_class[class_id]['false_positives'][threshold] += 1
                
                # False negatives
                for gt in ground_truth:
                    class_id = gt['class_id']
                    results_per_class[class_id]['total_ground_truth'] += 1
                
                unused_gt = len(ground_truth) - len(used_gt)
                for gt_idx, gt in enumerate(ground_truth):
                    if gt_idx not in used_gt:
                        class_id = gt['class_id']
                        results_per_class[class_id]['false_negatives'][threshold] += 1
    
    # Compute metrics
    print("\n📊 COMPUTING METRICS...")
    
    metrics = {}
    
    # Mean IoU
    if all_ious:
        metrics['mean_IoU'] = {
            'overall': float(np.mean(all_ious)),
            'τIoU=0.5': float(np.mean([iou for iou in all_ious if iou >= 0.5])) if any(iou >= 0.5 for iou in all_ious) else 0.0,
            'τIoU=0.7': float(np.mean([iou for iou in all_ious if iou >= 0.7])) if any(iou >= 0.7 for iou in all_ious) else 0.0,
            'τIoU=0.9': float(np.mean([iou for iou in all_ious if iou >= 0.9])) if any(iou >= 0.9 for iou in all_ious) else 0.0
        }
    else:
        metrics['mean_IoU'] = {'overall': 0.0, 'τIoU=0.5': 0.0, 'τIoU=0.7': 0.0, 'τIoU=0.9': 0.0}
    
    # mAP calculations
    for metric_name, thresholds in iou_thresholds.items():
        aps_per_class = []
        
        for class_id in results_per_class:
            class_data = results_per_class[class_id]
            
            # Compute AP for each threshold and average
            aps_for_thresholds = []
            
            for threshold in thresholds:
                tp = class_data['true_positives'][threshold]
                fp = class_data['false_positives'][threshold]
                fn = class_data['false_negatives'][threshold]
                
                if tp + fp + fn > 0:
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    
                    # Simplified AP (without full P-R curve)
                    ap = precision * recall if precision > 0 and recall > 0 else 0
                    aps_for_thresholds.append(ap)
                else:
                    aps_for_thresholds.append(0)
            
            if aps_for_thresholds:
                aps_per_class.append(np.mean(aps_for_thresholds))
        
        metrics[metric_name] = float(np.mean(aps_per_class)) if aps_per_class else 0.0
    
    # Add class-wise details
    metrics['per_class_results'] = {}
    for class_id, class_data in results_per_class.items():
        class_name = class_names.get(class_id, f"class_{class_id}")
        
        metrics['per_class_results'][class_name] = {
            'total_ground_truth': class_data['total_ground_truth'],
            'mean_iou': float(np.mean(class_data['ious'])) if class_data['ious'] else 0.0,
            'true_positives_0.5': class_data['true_positives'][0.5],
            'false_positives_0.5': class_data['false_positives'][0.5],
            'false_negatives_0.5': class_data['false_negatives'][0.5]
        }
    
    return metrics

def save_results(metrics, model_path, output_dir="results"):
    """Saves results to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare metadata
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_path': model_path,
        'test_dataset': 'yolo_dataset/test',
        'metrics': metrics,
        'summary': {
            'mAP@0.50:0.95': metrics.get('mAP@0.50:0.95', 0.0),
            'mAP@0.50': metrics.get('mAP@0.50', 0.0),
            'mean_IoU_overall': metrics.get('mean_IoU', {}).get('overall', 0.0),
            'mean_IoU_τ0.5': metrics.get('mean_IoU', {}).get('τIoU=0.5', 0.0),
            'mean_IoU_τ0.7': metrics.get('mean_IoU', {}).get('τIoU=0.7', 0.0),
            'mean_IoU_τ0.9': metrics.get('mean_IoU', {}).get('τIoU=0.9', 0.0)
        }
    }
    
    # Save to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"yolov9_evaluation_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved: {filepath}")
    return filepath

def print_results_summary(metrics):
    """Displays a summary of results."""
    print("\n🎉 RESULTS SUMMARY")
    print("=" * 50)
    
    print("📊 MAIN METRICS:")
    print(f"   mAP@0.50:0.95: {metrics.get('mAP@0.50:0.95', 0.0):.3f}")
    print(f"   mAP@0.50:     {metrics.get('mAP@0.50', 0.0):.3f}")
    
    mean_iou = metrics.get('mean_IoU', {})
    print("\n📊 MEAN IoU:")
    print(f"   Overall:      {mean_iou.get('overall', 0.0):.3f}")
    print(f"   τIoU = 0.5:   {mean_iou.get('τIoU=0.5', 0.0):.3f}")
    print(f"   τIoU = 0.7:   {mean_iou.get('τIoU=0.7', 0.0):.3f}")
    print(f"   τIoU = 0.9:   {mean_iou.get('τIoU=0.9', 0.0):.3f}")
    
    print("\n📊 ADDITIONAL METRICS:")
    print(f"   τIoU = 0.5:   {metrics.get('τIoU=0.5', 0.0):.3f}")
    print(f"   τIoU = 0.7:   {metrics.get('τIoU=0.7', 0.0):.3f}")
    print(f"   τIoU = 0.9:   {metrics.get('τIoU=0.9', 0.0):.3f}")

def main():
    """Main function."""
    print("🚀 YOLOv9 MODEL TESTING")
    print("🎯 METRICS: mAP, mean IoU @ different τIoU")
    print("=" * 60)
    
    # Configuration
    device = setup_device()
    model_path = "fine_tuned_models/yolo_v9_best/weights/best.pt"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    # Load model
    print("\n🔧 Loading model...")
    model = load_yolov9_model(model_path, device)
    
    if model is None:
        print("❌ Failed to load model")
        return
    
    # Load class names
    class_names = load_class_names()
    
    # Load test data
    print("\n📸 Loading test data...")
    test_data = load_test_data()
    
    if not test_data:
        print("❌ No test data found")
        return
    
    # Evaluation
    print("\n🎯 Starting evaluation...")
    metrics = evaluate_model(model, test_data, class_names)
    
    # Save results
    results_file = save_results(metrics, model_path)
    
    # Display summary
    print_results_summary(metrics)
    
    print("\n🎉 TESTING COMPLETED")
    print(f"✅ Evaluated on {len(test_data)} images")
    print(f"📁 Results saved at: {results_file}")

if __name__ == "__main__":
    main()