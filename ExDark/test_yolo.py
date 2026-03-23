import os
import torch
from PIL import Image
import matplotlib.pyplot as plt

def setup_device():
    """
    Configures the compute device (GPU/CPU) based on CUDA_VISIBLE_DEVICES.

    Returns:
        str: Device name ('cuda:X' or 'cpu')
    """
    gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '1')
    
    if torch.cuda.is_available():
        print(f"🔧 CUDA_VISIBLE_DEVICES: {gpu_id}")
        print(f"🔧 Available GPU count: {torch.cuda.device_count()}")
        
        # If a specific GPU is set, use it
        if gpu_id != '-1' and torch.cuda.device_count() > 0:
            # When CUDA_VISIBLE_DEVICES=1, PyTorch may still see both GPUs,
            # but we want to use the one set in the environment variable
            if gpu_id.isdigit():
                gpu_index = int(gpu_id)
                # Check whether this GPU exists in PyTorch
                if gpu_index < torch.cuda.device_count():
                    device = f'cuda:{gpu_index}'
                else:
                    device = 'cuda:0'  # fallback
            else:
                device = 'cuda:0'
            
            print(f"🔧 Using device: {device}")
            
            try:
                # Set default GPU
                torch.cuda.set_device(device)
                print(f"🔧 GPU name: {torch.cuda.get_device_name()}")
            except Exception as e:
                print(f"⚠️ Could not set GPU {device}: {e}")
                device = 'cpu'
        else:
            device = 'cpu'
            print("🔧 Using CPU")
    else:
        device = 'cpu'
        print("⚠️ CUDA unavailable, using CPU")
    
    return device

def visualize_ultralytics_results(results, save_path):
    """
    Visualizes object detection results from an Ultralytics YOLO model.
    
    Args:
        results: Detection results from YOLO model
        save_path (str): Output path to save the image
        
    Returns:
        PIL.Image: Image with annotations
    """
    # Ultralytics has built-in visualization
    annotated_frame = results[0].plot()
    
    # Convert BGR -> RGB (OpenCV -> PIL)
    annotated_frame_rgb = annotated_frame[:, :, ::-1]
    
    # Save as PIL Image
    pil_image = Image.fromarray(annotated_frame_rgb)
    pil_image.save(save_path)
    print(f"💾 Result saved as: {save_path}")
    
    return pil_image

def create_comparison_plot(original_image, ultra_result, save_path):
    """
    Creates a side-by-side comparison of original image and detection result.
    
    Args:
        original_image (PIL.Image): Original input image
        ultra_result (PIL.Image): Detection result image
        save_path (str): Output path to save the comparison
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title("Original image", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Detection result
    axes[1].imshow(ultra_result)
    axes[1].set_title("YOLOv9 - Object detection", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Comparison saved as: {save_path}")
    plt.close()

def test_yolo_detection():
    """
    Main YOLOv9 test function.
    Loads a model from a local file, runs detection on bus.jpg,
    and saves results in the photos folder.
    """
    # Device setup
    device = setup_device()
    
    # File paths
    model_path = "weights/yolov9c.pt"
    input_image = "photos/bus.jpg"
    output_detection = "photos/bus_detection.jpg"
    output_comparison = "photos/bus_comparison.png"
    
    try:
        from ultralytics import YOLO
        
        print(f"\n🚀 Loading model from: {model_path}")
        
        # Check whether model exists
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return None, None, None, None
            
        # Check whether image exists
        if not os.path.exists(input_image):
            print(f"❌ Image not found: {input_image}")
            return None, None, None, None
        
        # Load model from local file
        model = YOLO(model_path)
        
        # Move model to device
        print(f"🔄 Moving model to {device}...")
        try:
            if device != 'cpu':
                model.to(device)
            print(f"✅ Model loaded successfully on {device}")
        except Exception as e:
            print(f"⚠️ GPU error ({e}) - switching to CPU")
            device = 'cpu'
            model.to(device)
            print(f"✅ Model loaded on {device}")
        
        print(f"🎯 Number of classes: {len(model.names)}")
        
        print(f"\n🖼️ Running detection on: {input_image}")
        
        # Load original image
        original_image = Image.open(input_image)
        
        # Run detection
        results = model(input_image)
        
        print("✅ Detection completed")
        print(f"📊 Found {len(results[0].boxes)} objects")
        
        # Visualize results
        result_image = visualize_ultralytics_results(results, output_detection)
        
        # Create comparison
        create_comparison_plot(original_image, result_image, output_comparison)
        
        return model, results, result_image, original_image
        
    except ImportError:
        print("❌ Ultralytics library is not installed")
        return None, None, None, None
    except Exception as e:
        print(f"❌ Error during detection: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

if __name__ == "__main__":
    print("🎯 YOLOv9 MODEL TEST - OBJECT DETECTION")
    print("=" * 50)
    
    model, results, result_image, original_image = test_yolo_detection()
    
    if model is not None:
        print("\n✅ TEST COMPLETED SUCCESSFULLY!")
        print("📁 Files saved in photos/:")
        print("  • bus_detection.jpg - detection result")
        print("  • bus_comparison.png - image comparison")
    else:
        print("\n❌ TEST FAILED")
        
        # Fallback - try CPU
        print("\n🔄 Trying to run on CPU...")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        model, results, result_image, original_image = test_yolo_detection()
        
        if model is not None:
            print("\n✅ TEST COMPLETED SUCCESSFULLY ON CPU!")