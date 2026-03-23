import os
import torch
import numpy as np
from PIL import Image
import glob
from tqdm import tqdm
from transformers import Swin2SRForImageSuperResolution, AutoImageProcessor

def setup_device():
    """Configures GPU/CPU."""
    if torch.cuda.is_available():
        device = 'cuda:1'  # Use the same GPU as for training
        print(f"✅ GPU: {torch.cuda.get_device_name(1)}")
    else:
        device = 'cpu'
        print("⚠️ Using CPU")
    return device

def load_finetuned_swin2sr(model_path, device='cpu'):
    """Loads FINE-TUNED Swin2SR from a local folder."""
    try:
        print(f"📥 Loading FINE-TUNED Swin2SR...")
        print(f"📁 Path: {model_path}")
        
        # Check whether folder exists
        if not os.path.exists(model_path):
            print(f"❌ Model folder does not exist: {model_path}")
            return None, None
        
        # Check required files - FIX for safetensors
        required_files = ['config.json', 'preprocessor_config.json']
        model_files = ['model.safetensors', 'pytorch_model.bin']  # Both formats
        
        # Check config files
        for file in required_files:
            file_path = os.path.join(model_path, file)
            if not os.path.exists(file_path):
                print(f"❌ Missing file: {file_path}")
                return None, None
            else:
                print(f"✅ {file}")
        
        # Check which model format exists
        model_file_found = None
        for model_file in model_files:
            file_path = os.path.join(model_path, model_file)
            if os.path.exists(file_path):
                model_file_found = model_file
                print(f"✅ {model_file}")
                break
        
        if model_file_found is None:
            print(f"❌ Missing model file (searched for: {model_files})")
            return None, None
        
        # Load processor and model from local folder
        processor = AutoImageProcessor.from_pretrained(model_path)
        model = Swin2SRForImageSuperResolution.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            local_files_only=True,  # IMPORTANT: local files only
            use_safetensors=True if model_file_found == 'model.safetensors' else False  # ADDED
        )
        
        # Move to device
        model.to(device)
        model.eval()
        
        # Check size
        param_count = sum(p.numel() for p in model.parameters())
        
        print(f"✅ Fine-tuned Swin2SR loaded on {device}")
        print(f"📊 Parameters: {param_count/1e6:.1f}M")
        print(f"🎯 Scale: x2 (fine-tuned)")
        print(f"🏆 Model type: FINE-TUNED")
        print(f"📁 Format: {model_file_found}")
        
        return model, processor
    
    except Exception as e:
        print(f"❌ Error loading fine-tuned model: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def load_pretrained_swin2sr(device='cpu'):
    """Loads PRETRAINED Swin2SR from Hugging Face (fallback)."""
    model_name = "caidas/swin2SR-lightweight-x2-64"
    
    try:
        print(f"📥 Loading PRETRAINED Swin2SR from Hugging Face...")
        print(f"🔗 Model: {model_name}")
        
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = Swin2SRForImageSuperResolution.from_pretrained(model_name)
        
        model.to(device)
        model.eval()
        
        param_count = sum(p.numel() for p in model.parameters())
        
        print(f"✅ Pretrained Swin2SR loaded on {device}")
        print(f"📊 Parameters: {param_count/1e6:.1f}M")
        print(f"🎯 Scale: x2 (pretrained)")
        print(f"🤗 Model type: PRETRAINED")
        
        return model, processor
    
    except Exception as e:
        print(f"❌ Error loading pretrained model: {e}")
        return None, None

def swin2sr_inference(model, processor, image, device='cpu'):
    """Runs Swin2SR x2 inference."""
    if model is None or processor is None:
        return None
    
    try:
        # Prepare input via processor
        inputs = processor(image, return_tensors="pt")
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Swin2SR inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extract reconstruction
        reconstructed = outputs.reconstruction
        
        # Convert tensor -> PIL
        output_array = reconstructed.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        
        # Normalization
        if output_array.min() < 0 or output_array.max() > 1:
            if output_array.min() >= -1 and output_array.max() <= 1:
                output_array = (output_array + 1) / 2  # [-1, 1] -> [0, 1]
            else:
                output_array = np.clip(output_array, 0, 1)
        
        output_array = np.clip(output_array, 0.0, 1.0)
        output_uint8 = (output_array * 255.0).round().astype(np.uint8)
        
        sr_image = Image.fromarray(output_uint8, mode='RGB')
        
        return sr_image
    
    except Exception as e:
        print(f"❌ Swin2SR inference error: {e}")
        return None

def double_swin2sr_inference(model, processor, image, device='cpu'):
    """Performs DOUBLE Swin2SR x2: x2 → x2 = x4 total."""
    if model is None or processor is None:
        return None
    
    try:
        # FIRST Swin2SR x2 pass
        first_sr = swin2sr_inference(model, processor, image, device)
        if first_sr is None:
            return None
        
        # SECOND Swin2SR x2 pass (x4 total)
        second_sr = swin2sr_inference(model, processor, first_sr, device)
        if second_sr is None:
            return None
        
        return second_sr
    
    except Exception as e:
        print(f"❌ Double Swin2SR error: {e}")
        return None

def find_lr_images(lr_dir):
    """Finds all LR images in a folder."""
    if not os.path.exists(lr_dir):
        print(f"❌ Folder does not exist: {lr_dir}")
        return []
    
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(lr_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(lr_dir, '**', ext), recursive=True))
    
    image_paths = sorted(image_paths)
    print(f"📁 Found {len(image_paths)} images in {lr_dir}")
    
    return image_paths

def process_lr_2x_images(model, processor, device, input_dir, output_dir, model_type="finetuned"):
    """Processes LR_2x images through a single Swin2SR x2 pass."""
    print(f"\n🔄 PROCESSING LR_2x → SR_2x ({model_type.upper()})")
    print("=" * 50)
    
    # Find images
    lr_images = find_lr_images(input_dir)
    
    if not lr_images:
        print(f"❌ No images found in {input_dir}")
        return
    
    # Create output folder
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output folder: {output_dir}")
    
    # Process each image
    successful = 0
    failed = 0
    
    for img_path in tqdm(lr_images, desc=f"🚀 {model_type} x2"):
        try:
            # Load LR image
            lr_image = Image.open(img_path).convert('RGB')
            original_size = lr_image.size
            
            # Run Swin2SR x2 (single pass)
            sr_image = swin2sr_inference(model, processor, lr_image, device)
            
            if sr_image is not None:
                filename = os.path.basename(img_path)
                output_path = os.path.join(output_dir, filename)
                sr_image.save(output_path)
                successful += 1
                
                if successful % 10 == 1:
                    print(f"✅ {successful}: {filename} | {original_size} → {sr_image.size}")
            else:
                print(f"❌ Error for: {os.path.basename(img_path)}")
                failed += 1
        
        except Exception as e:
            print(f"❌ Processing error {img_path}: {e}")
            failed += 1
    
    print(f"\n📊 LR_2x SUMMARY ({model_type.upper()}):")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Results in: {output_dir}")

def process_lr_4x_images(model, processor, device, input_dir, output_dir, model_type="finetuned"):
    """Processes LR_4x images through double Swin2SR x2."""
    print(f"\n🔄 PROCESSING LR_4x → SR_4x ({model_type.upper()} DOUBLE)")
    print("=" * 50)
    
    # Find images
    lr_images = find_lr_images(input_dir)
    
    if not lr_images:
        print(f"❌ No images found in {input_dir}")
        return
    
    # Create output folder
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output folder: {output_dir}")
    
    # Process each image
    successful = 0
    failed = 0
    
    for img_path in tqdm(lr_images, desc=f"🚀 {model_type} x2→x2"):
        try:
            # Load LR image
            lr_image = Image.open(img_path).convert('RGB')
            original_size = lr_image.size
            
            # Run double Swin2SR x2 (double pass = x4 total)
            sr_image = double_swin2sr_inference(model, processor, lr_image, device)
            
            if sr_image is not None:
                filename = os.path.basename(img_path)
                output_path = os.path.join(output_dir, filename)
                sr_image.save(output_path)
                successful += 1
                
                if successful % 10 == 1:
                    print(f"✅ {successful}: {filename} | {original_size} → {sr_image.size}")
            else:
                print(f"❌ Error for: {os.path.basename(img_path)}")
                failed += 1
        
        except Exception as e:
            print(f"❌ Processing error {img_path}: {e}")
            failed += 1
    
    print(f"\n📊 LR_4x SUMMARY ({model_type.upper()}):")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Results in: {output_dir}")

def verify_model_and_folders(model_path):
    """Checks whether model and input folders exist."""
    print("🔍 CHECKING MODEL AND FOLDERS")
    print("=" * 50)
    
    # Check model
    if os.path.exists(model_path):
        # FIX: Check both model formats
        required_files = ['config.json', 'preprocessor_config.json']
        model_files = ['model.safetensors', 'pytorch_model.bin']
        
        model_ok = True
        
        # Check config files
        for file in required_files:
            file_path = os.path.join(model_path, file)
            if os.path.exists(file_path):
                print(f"✅ {file}")
            else:
                print(f"❌ {file} - MISSING!")
                model_ok = False
        
        # Check which model format exists
        model_file_found = False
        for model_file in model_files:
            file_path = os.path.join(model_path, model_file)
            if os.path.exists(file_path):
                print(f"✅ {model_file}")
                model_file_found = True
                break
        
        if not model_file_found:
            print(f"❌ Missing model file (searched for: {model_files})")
            model_ok = False
    else:
        print(f"❌ Model folder does not exist: {model_path}")
        model_ok = False
    
    # Check input folders
    required_folders = [
        "swin2_sr_dataset/test/LR_2x",
        "swin2_sr_dataset/test/LR_4x"
    ]
    
    folders_ok = True
    for folder in required_folders:
        if os.path.exists(folder):
            image_count = len(find_lr_images(folder))
            print(f"✅ {folder} - {image_count} images")
        else:
            print(f"❌ {folder} - MISSING!")
            folders_ok = False
    
    return model_ok, folders_ok

def main():
    """Main function with model selection."""
    print("🚀 PROCESSING SWIN2_SR DATASET WITH FINE-TUNED SWIN2SR")
    print("🎯 LR_2x → Single Fine-tuned Swin2SR x2")
    print("🎯 LR_4x → Double Fine-tuned Swin2SR x2")
    print("🏆 Model: FINE-TUNED on swin2_sr dataset")
    print("=" * 60)
    
    # Path configuration
    finetuned_model_path = "fine_tuned_swin2sr_2x/best_model"  # ← YOUR FINE-TUNED MODEL
    device = setup_device()
    
    # Data paths
    lr_2x_dir = "swin2_sr_dataset/test/LR_2x"
    lr_4x_dir = "swin2_sr_dataset/test/LR_4x"
    sr_2x_dir = "swin2_sr_dataset/test/SR_2x_finetuned"  # New folder for fine-tuned
    sr_4x_dir = "swin2_sr_dataset/test/SR_4x_finetuned"  # New folder for fine-tuned
    
    # Check model and folders
    model_ok, folders_ok = verify_model_and_folders(finetuned_model_path)
    
    if not folders_ok:
        print("❌ Missing required input folders")
        return
    
    # Load model (fine-tuned or fallback)
    if model_ok:
        print(f"\n🏆 Loading FINE-TUNED model...")
        model, processor = load_finetuned_swin2sr(finetuned_model_path, device)
        model_type = "finetuned"
        
        if model is None:
            print("⚠️ Failed to load fine-tuned model, switching to pretrained")
            model, processor = load_pretrained_swin2sr(device)
            model_type = "pretrained"
            # Change output folders
            sr_2x_dir = "swin2_sr_dataset/test/SR_2x_pretrained"
            sr_4x_dir = "swin2_sr_dataset/test/SR_4x_pretrained"
    else:
        print("⚠️ Fine-tuned model unavailable, using pretrained")
        model, processor = load_pretrained_swin2sr(device)
        model_type = "pretrained"
        # Change output folders
        sr_2x_dir = "swin2_sr_dataset/test/SR_2x_pretrained"
        sr_4x_dir = "swin2_sr_dataset/test/SR_4x_pretrained"
    
    if model is None or processor is None:
        print("❌ Failed to load any model")
        return
    
    # Process images
    process_lr_2x_images(model, processor, device, lr_2x_dir, sr_2x_dir, model_type)
    process_lr_4x_images(model, processor, device, lr_4x_dir, sr_4x_dir, model_type)
    
    # Summary
    print(f"\n🎉 PROCESSING COMPLETED")
    print("=" * 60)
    print(f"🏆 Model used: {model_type.upper()}")
    print(f"📁 Results:")
    print(f"   SR_2x: {sr_2x_dir}")
    print(f"   SR_4x: {sr_4x_dir}")
    print(f"\n🎯 STRATEGY:")
    print(f"   LR_2x → {model_type} Swin2SR x2 → SR_2x")
    print(f"   LR_4x → {model_type} Swin2SR x2→x2 → SR_4x")
    
    if model_type == "finetuned":
        print(f"🏆 YOU ARE USING A FINE-TUNED MODEL!")
        print(f"📊 Model trained on swin2_sr dataset")

if __name__ == "__main__":
    main()