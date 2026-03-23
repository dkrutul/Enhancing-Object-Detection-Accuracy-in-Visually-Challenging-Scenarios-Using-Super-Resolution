import os
import glob
import shutil
import random
from pathlib import Path
import yaml
from collections import defaultdict
from PIL import Image

def setup_yolo_directory_structure(output_dir):
    """
    Creates the folder structure for a YOLO dataset.
    
    Args:
        output_dir (str): Path to the main output directory
    """
    print("📁 Creating YOLO folder structure...")
    
    # Main folder
    yolo_root = Path(output_dir)
    
    # Folder structure
    folders_to_create = [
        yolo_root / "images" / "train",
        yolo_root / "images" / "val", 
        yolo_root / "images" / "test",
        yolo_root / "labels" / "train",
        yolo_root / "labels" / "val",
        yolo_root / "labels" / "test"
    ]
    
    # Create all folders
    for folder in folders_to_create:
        folder.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {folder}")
    
    return yolo_root

def find_matching_annotation(image_path, annotation_dir):
    """
    Finds a matching annotation file for an image.
    
    Args:
        image_path (str): Path to the image
        annotation_dir (str): Annotation folder for the class
        
    Returns:
        str: Path to annotation file, or None if not found
    """
    image_name = Path(image_path).name  # full filename with extension
    
    # ExDark annotation pattern: image.jpg → image.jpg.txt
    annotation_name = f"{image_name}.txt"
    annotation_path = os.path.join(annotation_dir, annotation_name)
    
    if os.path.exists(annotation_path):
        return annotation_path
    
    # Fallback: try filename without extension
    image_stem = Path(image_path).stem
    fallback_annotation = os.path.join(annotation_dir, f"{image_stem}.txt")
    
    if os.path.exists(fallback_annotation):
        return fallback_annotation
    
    return None

def parse_exdark_annotation(annotation_path, image_width, image_height, class_mapping):
    """
    Parses ExDark annotation and converts it to YOLO format.
    
    ExDark format: "Boat 28 50 229 216 0 0 0 0 0 0 0"
    - Class_name x y w h [additional_fields...]
    - x,y = top-left corner (pixels)
    - w,h = width,height (pixels)
    
    YOLO format: "class_id x_center y_center width height"
    - All coordinates normalized to [0,1]
    - x_center, y_center = bbox center
    
    Args:
        annotation_path (str): Path to ExDark annotation file
        image_width (int): Image width in pixels
        image_height (int): Image height in pixels
        class_mapping (dict): Mapping from class names to IDs
        
    Returns:
        list: List of YOLO bboxes [(class_id, x_center, y_center, width, height), ...]
    """
    yolo_bboxes = []
    
    try:
        with open(annotation_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            
            # Skip comments and empty lines
            if line.startswith('%') or not line:
                continue
            
            parts = line.split()
            if len(parts) < 5:
                continue
            
            # Parse ExDark format
            class_name = parts[0]
            x = int(parts[1])      # top-left x (pixels)
            y = int(parts[2])      # top-left y (pixels)  
            w = int(parts[3])      # width (pixels)
            h = int(parts[4])      # height (pixels)
            
            # Check whether class is supported
            if class_name not in class_mapping:
                print(f"⚠️ Unknown class: {class_name} in {annotation_path}")
                continue
            
            # Check whether bbox is inside image bounds
            if x < 0 or y < 0 or x + w > image_width or y + h > image_height:
                print(f"⚠️ Bbox outside image bounds: {class_name} in {annotation_path}")
                continue
            
            if w <= 0 or h <= 0:
                print(f"⚠️ Invalid bbox dimensions: {class_name} in {annotation_path}")
                continue
            
            # Convert ExDark → YOLO
            class_id = class_mapping[class_name]
            
            # Convert corner coordinates to center coordinates
            x_center_px = x + w / 2
            y_center_px = y + h / 2
            
            # Normalize to [0,1]
            x_center = x_center_px / image_width
            y_center = y_center_px / image_height
            norm_width = w / image_width
            norm_height = h / image_height
            
            # Check whether normalized coordinates are in [0,1]
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                   0 <= norm_width <= 1 and 0 <= norm_height <= 1):
                print(f"⚠️ Invalid normalized coordinates: {class_name} in {annotation_path}")
                continue
            
            yolo_bboxes.append((class_id, x_center, y_center, norm_width, norm_height))
    
    except Exception as e:
        print(f"❌ Parsing error {annotation_path}: {e}")
        return []
    
    return yolo_bboxes

def validate_image_annotation_pair(image_path, annotation_path, class_mapping):
    """
    Validates whether an image-annotation pair is correct.
    
    Args:
        image_path (str): Path to image file
        annotation_path (str): Path to annotation file
        class_mapping (dict): Class name mapping
        
    Returns:
        tuple: (is_valid, yolo_bboxes, image_size)
    """
    try:
        # Check whether files exist
        if not os.path.exists(image_path):
            return False, [], None
        if not os.path.exists(annotation_path):
            return False, [], None
        
        # Load image and get size
        with Image.open(image_path) as img:
            image_width, image_height = img.size
        
        # Check image size validity
        if image_width <= 0 or image_height <= 0:
            return False, [], None
        
        # Parse annotation
        yolo_bboxes = parse_exdark_annotation(
            annotation_path, image_width, image_height, class_mapping
        )
        
        # Check whether there are valid bboxes
        if not yolo_bboxes:
            return False, [], (image_width, image_height)
        
        return True, yolo_bboxes, (image_width, image_height)
    
    except Exception as e:
        print(f"❌ Validation error {image_path}: {e}")
        return False, [], None

def collect_valid_pairs(exdark_path, exdark_anno_path, class_mapping):
    """
    Collects all valid image-annotation pairs across all classes.
    
    Args:
        exdark_path (str): Path to ExDark folder
        exdark_anno_path (str): Path to ExDark_Anno folder
        class_mapping (dict): Mapping from class names to IDs
        
    Returns:
        dict: {class_name: [(image_path, annotation_path, yolo_bboxes, image_size), ...]}
    """
    print("🔍 Collecting valid image-annotation pairs...")
    
    # ExDark classes
    exdark_classes = list(class_mapping.keys())
    
    valid_pairs_by_class = {}
    total_images = 0
    total_valid_pairs = 0
    skipped_by_class = defaultdict(int)
    bbox_stats = defaultdict(int)
    
    for class_name in exdark_classes:
        print(f"\n📂 Processing class: {class_name}")
        
        # Class directories
        class_images_dir = os.path.join(exdark_path, class_name)
        class_annotations_dir = os.path.join(exdark_anno_path, class_name)
        
        if not os.path.exists(class_images_dir):
            print(f"⚠️ Missing image folder: {class_images_dir}")
            continue
            
        if not os.path.exists(class_annotations_dir):
            print(f"⚠️ Missing annotation folder: {class_annotations_dir}")
            continue
        
        # Find all images in class
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        images = []
        
        for ext in image_extensions:
            images.extend(glob.glob(os.path.join(class_images_dir, ext)))
            # Also search subfolders
            images.extend(glob.glob(os.path.join(class_images_dir, '**', ext), recursive=True))
        
        print(f"   📸 Found {len(images)} images")
        total_images += len(images)
        
        # Match image-annotation pairs
        valid_pairs = []
        
        for image_path in images:
            # Find matching annotation
            annotation_path = find_matching_annotation(image_path, class_annotations_dir)
            
            if annotation_path:
                # Validate pair
                is_valid, yolo_bboxes, image_size = validate_image_annotation_pair(
                    image_path, annotation_path, class_mapping
                )
                
                if is_valid:
                    valid_pairs.append((image_path, annotation_path, yolo_bboxes, image_size))
                    bbox_stats[class_name] += len(yolo_bboxes)
                else:
                    skipped_by_class[class_name] += 1
            else:
                skipped_by_class[class_name] += 1
        
        valid_pairs_by_class[class_name] = valid_pairs
        total_valid_pairs += len(valid_pairs)
        
        print(f"   ✅ Valid pairs: {len(valid_pairs)}")
        print(f"   📦 Bounding boxes: {bbox_stats[class_name]}")
        print(f"   ❌ Skipped: {skipped_by_class[class_name]}")
    
    # Summary
    print(f"\n📊 COLLECTION SUMMARY:")
    print(f"   📸 Total images: {total_images}")
    print(f"   ✅ Valid pairs: {total_valid_pairs}")
    print(f"   📦 Total bounding boxes: {sum(bbox_stats.values())}")
    print(f"   ❌ Total skipped: {total_images - total_valid_pairs}")
    
    # Per-class bbox stats
    print(f"\n📈 BOUNDING BOXES PER CLASS:")
    for class_name in sorted(bbox_stats.keys()):
        print(f"   📦 {class_name}: {bbox_stats[class_name]} boxes")
    
    return valid_pairs_by_class

def stratified_split(valid_pairs_by_class, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    Performs stratified train/val/test split for each class independently.
    
    Args:
        valid_pairs_by_class (dict): Image-annotation pairs per class
        train_ratio (float): Training set ratio
        val_ratio (float): Validation set ratio
        test_ratio (float): Test set ratio
        
    Returns:
        tuple: (splits, split_stats)
    """
    print(f"\n🎯 Stratified split {train_ratio:.0%}:{val_ratio:.0%}:{test_ratio:.0%}")
    
    splits = {'train': [], 'val': [], 'test': []}
    split_stats = defaultdict(lambda: defaultdict(int))
    
    for class_name, pairs in valid_pairs_by_class.items():
        if not pairs:
            print(f"⚠️ No data for class {class_name}")
            continue
        
        print(f"\n📂 Splitting class {class_name}: {len(pairs)} pairs")
        
        # Shuffle for randomness
        pairs_copy = pairs.copy()
        random.shuffle(pairs_copy)
        
        # Compute split boundaries
        total = len(pairs_copy)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))
        
        # Split
        train_pairs = pairs_copy[:train_end]
        val_pairs = pairs_copy[train_end:val_end]
        test_pairs = pairs_copy[val_end:]
        
        # Add to global splits with class info
        for pair in train_pairs:
            splits['train'].append((*pair, class_name))
        for pair in val_pairs:
            splits['val'].append((*pair, class_name))
        for pair in test_pairs:
            splits['test'].append((*pair, class_name))
        
        # Stats
        split_stats[class_name]['train'] = len(train_pairs)
        split_stats[class_name]['val'] = len(val_pairs)
        split_stats[class_name]['test'] = len(test_pairs)
        split_stats[class_name]['total'] = total
        
        print(f"   🚂 Train: {len(train_pairs)}")
        print(f"   ✅ Val: {len(val_pairs)}")
        print(f"   🧪 Test: {len(test_pairs)}")
    
    # Global summary
    print(f"\n📊 SPLIT SUMMARY:")
    print(f"   🚂 Train: {len(splits['train'])} pairs")
    print(f"   ✅ Val: {len(splits['val'])} pairs")
    print(f"   🧪 Test: {len(splits['test'])} pairs")
    print(f"   📦 Total: {sum(len(s) for s in splits.values())} pairs")
    
    return splits, split_stats

def copy_to_yolo_structure(splits, yolo_root):
    """
    Copies files into YOLO structure with proper naming.
    
    Args:
        splits (dict): train/val/test split data
        yolo_root (Path): YOLO dataset root folder
    """
    print(f"\n📋 Copying files into YOLO structure...")
    
    copy_stats = defaultdict(int)
    bbox_count_stats = defaultdict(int)
    
    for split_name, pairs in splits.items():
        print(f"\n📁 Copying {split_name} split ({len(pairs)} files)...")
        
        images_dir = yolo_root / "images" / split_name
        labels_dir = yolo_root / "labels" / split_name
        
        for i, (image_path, annotation_path, yolo_bboxes, image_size, class_name) in enumerate(pairs):
            try:
                # Generate unique filenames
                base_name = f"{class_name.lower()}_{i:04d}"
                
                # Copy image
                image_ext = Path(image_path).suffix
                new_image_name = f"{base_name}{image_ext}"
                new_image_path = images_dir / new_image_name
                shutil.copy2(image_path, new_image_path)
                
                # Save YOLO annotation
                new_label_name = f"{base_name}.txt"
                new_label_path = labels_dir / new_label_name
                save_yolo_annotation(new_label_path, yolo_bboxes)
                
                copy_stats[split_name] += 1
                bbox_count_stats[split_name] += len(yolo_bboxes)
                
                if (i + 1) % 100 == 0:
                    print(f"   📋 Copied {i + 1}/{len(pairs)} files...")
                    
            except Exception as e:
                print(f"❌ Copy error {image_path}: {e}")
    
    # Copy summary
    print(f"\n✅ COPYING COMPLETED:")
    for split_name in ['train', 'val', 'test']:
        if split_name in copy_stats:
            print(f"   📁 {split_name}: {copy_stats[split_name]} images, {bbox_count_stats[split_name]} bboxes")

def save_yolo_annotation(label_path, yolo_bboxes):
    """
    Saves annotations in YOLO format.
    
    Args:
        label_path (Path): Path to label file
        yolo_bboxes (list): YOLO bboxes (class_id, x_center, y_center, width, height)
    """
    with open(label_path, 'w') as f:
        for class_id, x_center, y_center, width, height in yolo_bboxes:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def create_dataset_yaml(yolo_root, class_names):
    """
    Creates dataset.yaml file for YOLOv9.
    
    Args:
        yolo_root (Path): Dataset root folder
        class_names (list): List of class names
    """
    print(f"\n📄 Creating dataset.yaml...")
    
    # Paths relative to dataset.yaml
    dataset_config = {
        'path': str(yolo_root.absolute()),
        'train': 'images/train',
        'val': 'images/val', 
        'test': 'images/test',
        'nc': len(class_names),
        'names': class_names
    }
    
    yaml_path = yolo_root / "dataset.yaml"
    
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Created: {yaml_path}")
    
    # Show content
    print(f"\n📄 dataset.yaml content:")
    with open(yaml_path, 'r') as f:
        print(f.read())

def save_split_report(yolo_root, split_stats, copy_stats, class_names):
    """
    Saves a detailed dataset split report.
    
    Args:
        yolo_root (Path): Dataset root folder
        split_stats (dict): Per-class split statistics
        copy_stats (dict): File copy statistics
        class_names (list): List of class names
    """
    print(f"\n📊 Saving split report...")
    
    report_path = yolo_root / "split_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("EXDARK → YOLO DATASET CONVERSION REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("DATASET INFO:\n")
        f.write("-" * 15 + "\n")
        f.write(f"ExDark classes: {len(class_names)}\n")
        f.write(f"Classes: {', '.join(class_names)}\n")
        f.write("Split ratio: 60% train, 20% val, 20% test\n\n")
        
        # Global statistics
        f.write("GLOBAL STATISTICS:\n")
        f.write("-" * 20 + "\n")
        total_train = sum(stats['train'] for stats in split_stats.values())
        total_val = sum(stats['val'] for stats in split_stats.values())
        total_test = sum(stats['test'] for stats in split_stats.values())
        total_all = total_train + total_val + total_test
        
        f.write(f"Train: {total_train} images ({total_train/total_all*100:.1f}%)\n")
        f.write(f"Val: {total_val} images ({total_val/total_all*100:.1f}%)\n")
        f.write(f"Test: {total_test} images ({total_test/total_all*100:.1f}%)\n")
        f.write(f"Total: {total_all} images\n\n")
        
        # Per-class statistics
        f.write("PER-CLASS STATISTICS:\n")
        f.write("-" * 25 + "\n")
        f.write(f"{'Class':<12} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8} {'%Train':<8} {'%Val':<8} {'%Test':<8}\n")
        f.write("-" * 80 + "\n")
        
        for class_name in class_names:
            if class_name in split_stats:
                stats = split_stats[class_name]
                total_class = stats['total']
                if total_class > 0:
                    train_pct = stats['train'] / total_class * 100
                    val_pct = stats['val'] / total_class * 100
                    test_pct = stats['test'] / total_class * 100
                    
                    f.write(f"{class_name:<12} {stats['train']:<8} {stats['val']:<8} "
                           f"{stats['test']:<8} {total_class:<8} {train_pct:<8.1f} "
                           f"{val_pct:<8.1f} {test_pct:<8.1f}\n")
                else:
                    f.write(f"{class_name:<12} {'0':<8} {'0':<8} {'0':<8} {'0':<8} "
                           f"{'0.0':<8} {'0.0':<8} {'0.0':<8}\n")
    
    print(f"✅ Report saved: {report_path}")

def main():
    """Main preprocessing function for ExDark → YOLO conversion."""
    print("🚀 PREPROCESSING EXDARK → YOLO DATASET")
    print("📋 ExDark format → YOLO format conversion")
    print("=" * 60)
    
    # Path configuration
    exdark_path = "ExDark"
    exdark_anno_path = "ExDark_Annno"
    output_dir = "yolo_dataset"
    
    # Check whether folders exist
    if not os.path.exists(exdark_path):
        print(f"❌ Missing ExDark folder: {exdark_path}")
        return
        
    if not os.path.exists(exdark_anno_path):
        print(f"❌ Missing ExDark_Annno folder: {exdark_anno_path}")
        return
    
    # Settings
    random.seed(42)  # For reproducibility
    
    # ExDark classes (same as original dataset)
    class_names = [
        'Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat',
        'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table'
    ]
    
    # Class-to-ID mapping (0-11)
    class_mapping = {name: i for i, name in enumerate(class_names)}
    
    print(f"📂 ExDark source: {exdark_path}")
    print(f"📝 Annotation source: {exdark_anno_path}")
    print(f"🎯 YOLO output: {output_dir}")
    print(f"🏷️ Number of classes: {len(class_names)}")
    print(f"🏷️ Classes: {', '.join(class_names)}")
    
    # 1. Create YOLO folder structure
    yolo_root = setup_yolo_directory_structure(output_dir)
    
    # 2. Collect valid image-annotation pairs with ExDark → YOLO conversion
    valid_pairs_by_class = collect_valid_pairs(exdark_path, exdark_anno_path, class_mapping)
    
    if not any(valid_pairs_by_class.values()):
        print("❌ No valid image-annotation pairs found")
        return
    
    # 3. Perform stratified split
    splits, split_stats = stratified_split(valid_pairs_by_class)
    
    # 4. Copy files into YOLO structure
    copy_to_yolo_structure(splits, yolo_root)
    
    # 5. Create dataset.yaml
    create_dataset_yaml(yolo_root, class_names)
    
    # 6. Save report
    copy_stats = {split: len(pairs) for split, pairs in splits.items()}
    save_split_report(yolo_root, split_stats, copy_stats, class_names)
    
    # Final summary
    print(f"\n🎉 PREPROCESSING COMPLETED!")
    print(f"📁 YOLO dataset created at: {yolo_root}")
    print(f"🔧 Ready for YOLOv9 training with Ultralytics!")
    print(f"\n💡 Format conversion:")
    print("   📝 ExDark: 'Boat 28 50 229 216' → YOLO: '1 0.125 0.133 0.286 0.288'")
    print("   🔄 Pixel coordinates → Normalized coordinates")
    print("   📦 Corner-based → Center-based")
    print(f"\n🚀 Next steps:")

if __name__ == "__main__":
    main()