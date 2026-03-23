import os
import glob
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np

def parse_annotation_file(annotation_path):
    """
    Parses an ExDark annotation file in bbGt format.
    
    Args:
        annotation_path (str): Path to the annotation file
        
    Returns:
        list: List of dictionaries with bounding boxes
    """
    annotations = []
    
    if not os.path.exists(annotation_path):
        print(f"⚠️ Annotation file does not exist: {annotation_path}")
        return annotations
    
    try:
        with open(annotation_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            
            # Skip header and empty lines
            if line.startswith('%') or not line:
                continue
            
            parts = line.split()
            if len(parts) >= 5:
                class_name = parts[0]
                l = int(parts[1])  # left
                t = int(parts[2])  # top
                w = int(parts[3])  # width
                h = int(parts[4])  # height
                
                annotations.append({
                    'class': class_name,
                    'left': l,
                    'top': t,
                    'width': w,
                    'height': h,
                    'right': l + w,
                    'bottom': t + h
                })
                
    except Exception as e:
        print(f"❌ Parsing error {annotation_path}: {e}")
    
    return annotations

def find_annotation_file(annotations_path, image_filename):
    """
    Finds the annotation file corresponding to an image.
    
    Args:
        annotations_path (str): Path to the annotations folder
        image_filename (str): Image filename
        
    Returns:
        str or None: Path to annotation file or None if not found
    """
    # Different possible annotation extensions
    base_name = os.path.splitext(image_filename)[0]
    
    possible_names = [
        f"{image_filename}.txt",        # 2015_00001.png.txt
        f"{base_name}.txt",             # 2015_00001.txt
        f"{base_name}.xml",             # 2015_00001.xml (if present)
    ]
    
    # Search all possible locations
    for root, dirs, files in os.walk(annotations_path):
        for possible_name in possible_names:
            if possible_name in files:
                found_path = os.path.join(root, possible_name)
                print(f"   ✅ Found annotation: {possible_name}")
                return found_path
    
    print(f"   ❌ Annotation not found for: {image_filename}")
    print(f"      Searched for: {possible_names}")
    
    # Debug - show what is inside the folder
    for root, dirs, files in os.walk(annotations_path):
        if files:  # If folder contains files
            print(f"      Available files in {root}:")
            for f in sorted(files)[:5]:  # Show first 5
                print(f"        - {f}")
            if len(files) > 5:
                print(f"        ... and {len(files) - 5} more")
            break
    
    return None

def visualize_annotations_matplotlib(image, annotations, title=""):
    """
    Visualizes annotations using matplotlib.
    
    Args:
        image (PIL.Image): Input image
        annotations (list): List of annotations
        title (str): Plot title
    """
    # Convert image to RGB if it has alpha channel
    if image.mode in ('RGBA', 'LA', 'P'):
        # Create white background
        background = Image.new('RGB', image.size, (255, 255, 255))
        if image.mode == 'P':
            image = image.convert('RGBA')
        background.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
        image = background
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Display image
    ax.imshow(image)
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.axis('off')
    
    # Colors for different classes
    colors = {
        'Bicycle': 'red',
        'Boat': 'blue', 
        'Bottle': 'green',
        'Bus': 'orange',
        'Car': 'purple',
        'Cat': 'pink',
        'Chair': 'brown',
        'Cup': 'cyan',
        'Dog': 'yellow',
        'Motorbike': 'magenta',
        'People': 'lime',
        'Table': 'navy'
    }
    
    # Add bounding boxes
    for i, ann in enumerate(annotations):
        class_name = ann['class']
        left, top = ann['left'], ann['top']
        width, height = ann['width'], ann['height']
        
        # Select color
        color = colors.get(class_name, 'white')
        
        print(f"   🎨 Drawing bounding box: {class_name} [{left}, {top}, {width}, {height}] in {color}")
        
        # Create rectangle
        rect = Rectangle((left, top), width, height,
                        linewidth=3, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # Add label - no numbering
        label = class_name
        ax.text(left, max(0, top - 5), label, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
               fontsize=12, fontweight='bold', color='white')
    
    plt.tight_layout()
    return fig

def verify_first_images_annotations(images_path, annotations_path, output_dir="annotation_verification"):
    """
    Verifies annotations for the first image from each class.
    
    Args:
        images_path (str): Path to ExDark images folder
        annotations_path (str): Path to ExDark_Anno annotations folder
        output_dir (str): Output folder for visualizations
    """
    print(f"🔍 EXDARK ANNOTATION VERIFICATION")
    print(f"Images: {images_path}")
    print(f"Annotations: {annotations_path}")
    print("=" * 60)
    
    if not os.path.exists(images_path):
        print(f"❌ Images folder does not exist: {images_path}")
        return
    
    if not os.path.exists(annotations_path):
        print(f"❌ Annotations folder does not exist: {annotations_path}")
        return
    
    # Create output folder
    os.makedirs(output_dir, exist_ok=True)
    
    # ExDark classes
    exdark_classes = [
        'Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat',
        'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table'
    ]
    
    verified_count = 0
    
    for class_name in exdark_classes:
        print(f"\n📁 Verifying class: {class_name}")
        
        # Class folder paths
        class_images_path = os.path.join(images_path, class_name)
        class_annotations_path = os.path.join(annotations_path, class_name)
        
        if not os.path.exists(class_images_path):
            print(f"   ⚠️ Images folder does not exist: {class_images_path}")
            continue
            
        if not os.path.exists(class_annotations_path):
            print(f"   ⚠️ Annotations folder does not exist: {class_annotations_path}")
            continue
        
        # Find all images in class folder
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(glob.glob(os.path.join(class_images_path, '**', ext), recursive=True))
        
        if not image_files:
            print(f"   ⚠️ No images in folder: {class_images_path}")
            continue
        
        # Take first image
        first_image_path = sorted(image_files)[0]
        image_filename = os.path.basename(first_image_path)
        
        print(f"   🖼️ Image: {image_filename}")
        
        # Find matching annotation
        annotation_path = find_annotation_file(class_annotations_path, image_filename)
        
        if not annotation_path:
            continue
        
        try:
            # Load image
            with Image.open(first_image_path) as image:
                print(f"   📏 Image size: {image.size}")
                print(f"   🎨 Image mode: {image.mode}")
                
                # Parse annotations
                annotations = parse_annotation_file(annotation_path)
                print(f"   📊 Number of objects: {len(annotations)}")
                
                if annotations:
                    # Show annotation details
                    for i, ann in enumerate(annotations):
                        print(f"      {i+1}. {ann['class']}: [{ann['left']}, {ann['top']}, {ann['width']}, {ann['height']}]")
                    
                    # Create matplotlib visualization
                    title = f"{class_name} - {image_filename}\nSize: {image.size}, Objects: {len(annotations)}"
                    fig = visualize_annotations_matplotlib(image, annotations, title)
                    
                    # Save figure
                    plt_output_path = os.path.join(output_dir, f"{class_name}_verification.png")
                    fig.savefig(plt_output_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    
                    print(f"   💾 Saved: {plt_output_path}")
                    
                    verified_count += 1
                else:
                    print(f"   ⚠️ No valid annotations in file")
                    
        except Exception as e:
            print(f"   ❌ Processing error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✅ VERIFICATION COMPLETED")
    print(f"📊 Verified {verified_count}/{len(exdark_classes)} classes")
    print(f"📁 Results saved in: {output_dir}/")
    print(f"   • *_verification.png - annotation plots")

def create_annotation_summary(images_path, annotations_path):
    """
    Creates annotation summary for the full dataset.
    
    Args:
        images_path (str): Path to images folder
        annotations_path (str): Path to annotations folder
    """
    print(f"\n📊 ANNOTATION SUMMARY")
    print("=" * 40)
    
    exdark_classes = [
        'Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat',
        'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table'
    ]
    
    total_images = 0
    total_annotated = 0
    total_objects = 0
    class_object_counts = {}
    
    for class_name in exdark_classes:
        class_images_path = os.path.join(images_path, class_name)
        class_annotations_path = os.path.join(annotations_path, class_name)
        
        if os.path.exists(class_images_path) and os.path.exists(class_annotations_path):
            # Count images
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(glob.glob(os.path.join(class_images_path, '**', ext), recursive=True))
            
            # Count annotations
            annotation_files = glob.glob(os.path.join(class_annotations_path, '**', '*.txt'), recursive=True)
            
            class_objects = 0
            for ann_file in annotation_files:
                annotations = parse_annotation_file(ann_file)
                class_objects += len(annotations)
            
            total_images += len(image_files)
            total_annotated += len(annotation_files)
            total_objects += class_objects
            class_object_counts[class_name] = class_objects
            
            print(f"{class_name:>10}: {len(image_files):4d} images, {len(annotation_files):4d} annotations, {class_objects:4d} objects")
    
    print(f"\n📈 TOTAL STATISTICS:")
    print(f"   🖼️ Total images: {total_images}")
    print(f"   📝 Annotation files: {total_annotated}")
    print(f"   🎯 Total objects: {total_objects}")
    print(f"   📊 Average objects per image: {total_objects/max(total_annotated, 1):.2f}")

if __name__ == "__main__":
    print("🎯 EXDARK ANNOTATION VERIFICATION")
    print("=" * 50)
    
    # Folder paths
    images_path = "ExDark"            # Images folder
    annotations_path = "ExDark_Annno" # Annotations folder
    
    # Check whether folders exist
    if not os.path.exists(images_path):
        print(f"❌ Images folder not found: {images_path}")
        print("💡 Set the correct path in variable 'images_path'")
        exit(1)
    
    if not os.path.exists(annotations_path):
        print(f"❌ Annotations folder not found: {annotations_path}")
        print("💡 Set the correct path in variable 'annotations_path'")
        exit(1)
    
    # Run verification
    verify_first_images_annotations(images_path, annotations_path)
    
    # Create summary
    create_annotation_summary(images_path, annotations_path)
    
    print(f"\n🎉 DONE!")
    print(f"💡 Check folder 'annotation_verification/' for visualization results")