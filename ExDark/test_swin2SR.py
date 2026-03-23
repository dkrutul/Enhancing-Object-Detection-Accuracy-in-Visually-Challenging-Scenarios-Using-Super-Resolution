import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import glob
import requests
from transformers import Swin2SRForImageSuperResolution, AutoImageProcessor
from huggingface_hub import hf_hub_download

def setup_device():
    """Configures GPU/CPU."""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("⚠️ Using CPU")
    return device

def download_swin2sr_model():
    """Downloads the Swin2SR model from Hugging Face."""
    model_name = "caidas/swin2SR-lightweight-x2-64"
    
    print(f"📥 Downloading Swin2SR from Hugging Face...")
    print(f"🔗 Model: {model_name}")
    print(f"🎯 Scale: x2, lightweight version")
    
    try:
        # Download processor and model
        print("🔧 Loading image processor...")
        processor = AutoImageProcessor.from_pretrained(model_name)
        
        print("🔧 Loading pretrained model...")
        model = Swin2SRForImageSuperResolution.from_pretrained(model_name)
        
        print(f"✅ Swin2SR loaded from Hugging Face")
        return model, processor
    
    except Exception as e:
        print(f"❌ Error downloading from Hugging Face: {e}")
        print("💡 Check your internet connection")
        return None, None

def load_swin2sr_model(device='cpu'):
    """Loads PRETRAINED Swin2SR x2 model from Hugging Face."""
    try:
        # Download model and processor
        model, processor = download_swin2sr_model()
        
        if model is None or processor is None:
            print("❌ Failed to download Swin2SR")
            return None, None
        
        # Move to device
        model.to(device)
        model.eval()
        
        # Check model size
        param_count = sum(p.numel() for p in model.parameters())
        
        print(f"✅ Swin2SR PRETRAINED loaded on {device}")
        print(f"📊 Parameters: {param_count/1e6:.1f}M")
        print(f"🎯 Scale: x2 (lightweight)")
        print(f"🏢 Source: Hugging Face - caidas/swin2SR-lightweight-x2-64")
        print(f"🔄 Ready for double application: x2 → x2 = x4")
        
        return model, processor
    
    except Exception as e:
        print(f"❌ Error loading Swin2SR: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def swin2sr_inference(model, processor, image, device='cpu', debug_dir="debug"):
    """Runs Swin2SR x2 inference."""
    if model is None or processor is None:
        return None
    
    try:
        os.makedirs(debug_dir, exist_ok=True)
        
        print(f"🔍 Input image size: {image.size}")
        
        # Prepare input via processor
        inputs = processor(image, return_tensors="pt")
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        print(f"🔍 Processor input: {inputs['pixel_values'].shape}")
        print(f"🔍 Input range: {inputs['pixel_values'].min():.3f}-{inputs['pixel_values'].max():.3f}")
        
        # Swin2SR inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extract reconstruction
        reconstructed = outputs.reconstruction
        
        print(f"🔍 Output tensor: {reconstructed.shape}")
        print(f"🔍 Output range: {reconstructed.min():.3f}-{reconstructed.max():.3f}")
        
        # Verify x2 upscaling
        input_h, input_w = inputs['pixel_values'].shape[2], inputs['pixel_values'].shape[3]
        output_h, output_w = reconstructed.shape[2], reconstructed.shape[3]
        
        print(f"🔍 Input: {input_h}x{input_w}, Output: {output_h}x{output_w}")
        scale_h = output_h / input_h
        scale_w = output_w / input_w
        print(f"🔍 Scale factor: {scale_h:.1f}x{scale_w:.1f}")
        
        # Tensor -> PIL conversion
        output_array = reconstructed.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        
        # Normalization - Swin2SR can output in a different range
        if output_array.min() < 0 or output_array.max() > 1:
            print(f"⚠️ Denormalization required: {output_array.min():.3f} - {output_array.max():.3f}")
            # Check whether range is [-1, 1] or [0, 1]
            if output_array.min() >= -1 and output_array.max() <= 1:
                output_array = (output_array + 1) / 2  # [-1, 1] -> [0, 1]
            else:
                output_array = np.clip(output_array, 0, 1)  # Force [0, 1]
        
        output_array = np.clip(output_array, 0.0, 1.0)
        output_uint8 = (output_array * 255.0).round().astype(np.uint8)
        
        print(f"🔍 Final array: shape={output_uint8.shape}, range={output_uint8.min()}-{output_uint8.max()}")
        
        sr_image = Image.fromarray(output_uint8, mode='RGB')
        
        # Debug save
        debug_path = os.path.join(debug_dir, f"swin2sr_x2_{image.size[0]}x{image.size[1]}.png")
        sr_image.save(debug_path)
        print(f"💾 Debug Swin2SR: {debug_path}")
        
        return sr_image
    
    except Exception as e:
        print(f"❌ Swin2SR inference error: {e}")
        import traceback
        traceback.print_exc()
        return None

def double_swin2sr_inference(model, processor, image, device='cpu', debug_dir="debug"):
    """Performs DOUBLE Swin2SR x2: x2 → x2 = x4 total."""
    if model is None or processor is None:
        return None
    
    try:
        print("🔄 FIRST Swin2SR x2 iteration...")
        first_sr = swin2sr_inference(model, processor, image, device, debug_dir)
        if first_sr is None:
            return None
        
        print(f"✅ First iteration: {image.size} → {first_sr.size}")
        
        print("🔄 SECOND Swin2SR x2 iteration (total x4)...")
        second_sr = swin2sr_inference(model, processor, first_sr, device, debug_dir)
        if second_sr is None:
            return None
        
        print(f"✅ Second iteration: {first_sr.size} → {second_sr.size}")
        print(f"🎯 TOTAL: {image.size} → {second_sr.size} (x{second_sr.size[0]/image.size[0]:.1f})")
        
        return second_sr
    
    except Exception as e:
        print(f"❌ Double Swin2SR error: {e}")
        return None

def find_sample_images(exdark_path, max_per_class=1):
    """Finds sample images."""
    sample_images = {}
    
    exdark_classes = [
        'Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat',
        'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table'
    ]
    
    for class_name in exdark_classes:
        class_path = os.path.join(exdark_path, class_name)
        
        if not os.path.exists(class_path):
            continue
        
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(glob.glob(os.path.join(class_path, '**', ext), recursive=True))
        
        if image_files:
            sample_images[class_name] = image_files[:max_per_class]
            print(f"📁 {class_name}: {len(sample_images[class_name])} images")
    
    return sample_images

def create_comparison_set(image_path, model, processor, device):
    """Creates a comparison set with Swin2SR x2 (double application)."""
    print(f"\n🔍 Processing: {os.path.basename(image_path)}")
    
    # Debug folder
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    debug_dir = os.path.join("debug", base_name)
    os.makedirs(debug_dir, exist_ok=True)
    
    # Load original image
    original = Image.open(image_path).convert('RGB')
    original_size = original.size
    print(f"📐 Original size: {original_size}")
    
    # Save original
    original.save(os.path.join(debug_dir, "00_original.png"))
    
    # Degraded versions
    degraded_2x = original.resize((original_size[0] // 2, original_size[1] // 2), Image.BICUBIC)
    degraded_4x = original.resize((original_size[0] // 4, original_size[1] // 4), Image.BICUBIC)
    
    # Save degraded versions
    degraded_2x.save(os.path.join(debug_dir, "01_degraded_2x.png"))
    degraded_4x.save(os.path.join(debug_dir, "02_degraded_4x.png"))
    
    print(f"📉 Degraded 2x: {degraded_2x.size}")
    print(f"📉 Degraded 4x: {degraded_4x.size}")
    
    # Swin2SR outputs
    sr_2x = None
    sr_4x = None
    
    if model is not None and processor is not None:
        print("🚀 Running Swin2SR x2 (single)...")
        try:
            sr_temp = swin2sr_inference(model, processor, degraded_2x, device, debug_dir)
            if sr_temp:
                sr_2x = sr_temp.resize(original_size, Image.LANCZOS)
                print(f"✅ Swin2SR x2: {sr_2x.size}")
                sr_temp.save(os.path.join(debug_dir, "03_swin2sr_x2_raw.png"))
                sr_2x.save(os.path.join(debug_dir, "04_swin2sr_x2_resized.png"))
        except Exception as e:
            print(f"❌ Swin2SR x2 error: {e}")
        
        print("🚀 Running Swin2SR x2 → x2 (double)...")
        try:
            sr_temp = double_swin2sr_inference(model, processor, degraded_4x, device, debug_dir)
            if sr_temp:
                sr_4x = sr_temp.resize(original_size, Image.LANCZOS)
                print(f"✅ Swin2SR x4 (double): {sr_4x.size}")
                sr_temp.save(os.path.join(debug_dir, "05_swin2sr_x4_double_raw.png"))
                sr_4x.save(os.path.join(debug_dir, "06_swin2sr_x4_double_resized.png"))
        except Exception as e:
            print(f"❌ Swin2SR x4 (double) error: {e}")
    
    # Fallback to bicubic
    if sr_2x is None:
        sr_2x = degraded_2x.resize(original_size, Image.BICUBIC)
        sr_2x.save(os.path.join(debug_dir, "07_bicubic_2x_fallback.png"))
        print("⚠️ Swin2SR x2 fallback -> bicubic")
    
    if sr_4x is None:
        sr_4x = degraded_4x.resize(original_size, Image.BICUBIC)
        sr_4x.save(os.path.join(debug_dir, "08_bicubic_4x_fallback.png"))
        print("⚠️ Swin2SR x4 fallback -> bicubic")
    
    return {
        'original': original,
        'degraded_2x': degraded_2x,
        'degraded_4x': degraded_4x,
        'sr_2x': sr_2x,
        'sr_4x': sr_4x,
        'image_name': os.path.basename(image_path)
    }

def save_comparison_grid(comparison_set, output_dir="swin2sr_x2_double_comparison"):
    """Saves comparison grid."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    
    images = [
        (comparison_set['original'], 'Original', comparison_set['original'].size),
        (comparison_set['degraded_2x'], 'Degraded 2x', comparison_set['degraded_2x'].size),
        (comparison_set['degraded_4x'], 'Degraded 4x', comparison_set['degraded_4x'].size),
        (comparison_set['sr_2x'], 'Swin2SR x2', comparison_set['sr_2x'].size),
        (comparison_set['sr_4x'], 'Swin2SR x2→x2\n(Total x4)', comparison_set['sr_4x'].size)
    ]
    
    for i, (img, title, size) in enumerate(images):
        axes[i].imshow(img)
        axes[i].set_title(f"{title}\n{size[0]}×{size[1]}", fontsize=12, fontweight='bold')
        axes[i].axis('off')
    
    fig.suptitle(f"Swin2SR x2 DOUBLE APPLICATION: {comparison_set['image_name']}", 
                 fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    
    base_name = os.path.splitext(comparison_set['image_name'])[0]
    save_path = os.path.join(output_dir, f"{base_name}_swin2sr_x2_double.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Saved: {save_path}")

def save_individual_images(comparison_set, output_dir="swin2sr_x2_double_comparison"):
    """Saves individual images."""
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(comparison_set['image_name'])[0]
    
    comparison_set['original'].save(os.path.join(output_dir, f"{base_name}_1_original.png"))
    comparison_set['degraded_2x'].save(os.path.join(output_dir, f"{base_name}_2_degraded_2x.png"))
    comparison_set['degraded_4x'].save(os.path.join(output_dir, f"{base_name}_3_degraded_4x.png"))
    comparison_set['sr_2x'].save(os.path.join(output_dir, f"{base_name}_4_swin2sr_x2.png"))
    comparison_set['sr_4x'].save(os.path.join(output_dir, f"{base_name}_5_swin2sr_x2_double.png"))
    
    print(f"💾 Individual images saved for: {base_name}")

def main():
    """Main function."""
    print("🚀 COMPARISON: Swin2SR x2 DOUBLE APPLICATION")
    print("🏆 Swin2SR (2022) - Swin Transformer v2 SR")
    print("🎯 STRATEGY: x2 → x2 = x4 total")
    print("🤗 HUGGING FACE: caidas/swin2SR-lightweight-x2-64")
    print("=" * 60)
    
    # Configuration
    device = setup_device()
    exdark_path = "ExDark"
    
    if not os.path.exists(exdark_path):
        print(f"❌ ExDark folder not found: {exdark_path}")
        return
    
    # Load Swin2SR x2 model from Hugging Face
    print("\n🔧 Loading Swin2SR x2 from Hugging Face...")
    model, processor = load_swin2sr_model(device)
    
    if model is None or processor is None:
        print("❌ Failed to load Swin2SR")
        return
    
    # Find images
    print(f"\n📸 Searching for sample images...")
    sample_images = find_sample_images(exdark_path, max_per_class=1)
    
    if not sample_images:
        print("❌ No images found")
        return
    
    print(f"✅ Found images from {len(sample_images)} classes")
    
    # Process images
    processed_count = 0
    
    for class_name, image_paths in sample_images.items():
        for image_path in image_paths:
            try:
                print(f"\n{'='*60}")
                print(f"🎯 CLASS: {class_name}")
                
                comparison_set = create_comparison_set(image_path, model, processor, device)
                save_comparison_grid(comparison_set)
                save_individual_images(comparison_set)
                
                processed_count += 1
                
            except Exception as e:
                print(f"❌ Error for {image_path}: {e}")
                import traceback
                traceback.print_exc()
    
    # Summary
    print(f"\n{'='*60}")
    print(f"🎉 COMPLETED - Swin2SR x2 DOUBLE APPLICATION")
    print(f"✅ Processed images: {processed_count}")
    print(f"📁 Results in: swin2sr_x2_double_comparison/")
    print(f"🔍 Debug in: debug/")
    print(f"\n🎯 x2 DOUBLE STRATEGY:")
    print(f"   🔄 Single x2: degraded_2x → sr_2x")
    print(f"   🔄 Double x2: degraded_4x → sr_2x → sr_4x")
    print(f"   ✅ Total scaling: x4 via double x2")
    

if __name__ == "__main__":
    main()