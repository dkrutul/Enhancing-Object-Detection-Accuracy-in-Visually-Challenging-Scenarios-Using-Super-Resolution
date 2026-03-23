import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import Swin2SRForImageSuperResolution, AutoImageProcessor
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
import gc

class SRDataset(Dataset):
    """Dataset for fine-tuning Swin2SR with existing train/val folders."""
    
    def __init__(self, hr_dir, lr_dir, crop_size=128, scale_factor=2, mode='train'):
        """
        Args:
            hr_dir: Folder with High Resolution images (e.g., swin2_wr/train/HR)
            lr_dir: Folder with Low Resolution images (e.g., swin2_wr/train/LR_2x)
            crop_size: Patch size for training
            scale_factor: Super-resolution scale factor (2)
            mode: 'train' or 'val'
        """
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.crop_size = crop_size
        self.scale_factor = scale_factor
        self.mode = mode
        
        # IMPORTANT: Swin2SR requires crop_size to be divisible by 8
        if crop_size % 8 != 0:
            self.crop_size = ((crop_size // 8) + 1) * 8
            print(f"⚠️ Crop size adjusted to: {self.crop_size} (must be divisible by 8)")
        
        # Find HR-LR pairs
        self.image_pairs = self._find_image_pairs()
        
        print(f"📁 {mode.upper()} - HR: {hr_dir}")
        print(f"📁 {mode.upper()} - LR: {lr_dir}")
        print(f"📊 {mode.upper()} - Found {len(self.image_pairs)} image pairs")
        print(f"🎯 Crop size: {self.crop_size}x{self.crop_size}, Scale: x{scale_factor}")
    
    def _find_image_pairs(self):
        """Finds HR-LR image pairs with matching filenames."""
        pairs = []
        
        # Find all LR images (as reference)
        lr_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
        lr_files = []
        for ext in lr_extensions:
            lr_files.extend(glob.glob(os.path.join(self.lr_dir, ext)))
            lr_files.extend(glob.glob(os.path.join(self.lr_dir, '**', ext), recursive=True))
        
        for lr_path in lr_files:
            lr_name = os.path.splitext(os.path.basename(lr_path))[0]
            
            # Search for matching HR image
            for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                hr_path = os.path.join(self.hr_dir, lr_name + ext)
                if os.path.exists(hr_path):
                    pairs.append((hr_path, lr_path))
                    break
        
        return pairs
    
    def __len__(self):
        return len(self.image_pairs) * (8 if self.mode == 'train' else 1)  # More crops for train
    
    def _random_crop(self, hr_img, lr_img):
        """Randomly crops HR and matching LR patch with proper alignment."""
        hr_w, hr_h = hr_img.size
        lr_w, lr_h = lr_img.size
        
        # Check whether image sizes match expected scale
        expected_lr_w = hr_w // self.scale_factor
        expected_lr_h = hr_h // self.scale_factor
        
        if abs(lr_w - expected_lr_w) > 2 or abs(lr_h - expected_lr_h) > 2:
            # Resize LR to expected size
            lr_img = lr_img.resize((expected_lr_w, expected_lr_h), Image.BICUBIC)
            lr_w, lr_h = lr_img.size
        
        # Compute crop sizes - ENSURE DIVISIBLE BY 8
        lr_crop_size = self.crop_size // self.scale_factor
        
        # Check whether LR crop size is divisible by 8
        if lr_crop_size % 8 != 0:
            lr_crop_size = ((lr_crop_size // 8) + 1) * 8
            self.crop_size = lr_crop_size * self.scale_factor
            print(f"⚠️ LR crop size adjusted to: {lr_crop_size}")
        
        if lr_w < lr_crop_size or lr_h < lr_crop_size:
            # If image is too small, resize with margin to next multiple of 8
            scale = max(lr_crop_size / lr_w, lr_crop_size / lr_h) * 1.1
            new_lr_w = int(lr_w * scale)
            new_lr_h = int(lr_h * scale)
            
            # Ensure new sizes are divisible by 8
            new_lr_w = ((new_lr_w // 8) + 1) * 8
            new_lr_h = ((new_lr_h // 8) + 1) * 8
            
            lr_img = lr_img.resize((new_lr_w, new_lr_h), Image.BICUBIC)
            hr_img = hr_img.resize((new_lr_w * self.scale_factor, new_lr_h * self.scale_factor), Image.BICUBIC)
            lr_w, lr_h = lr_img.size
            hr_w, hr_h = hr_img.size
        
        # Random crop for train, center crop for val
        if self.mode == 'train':
            # Ensure crop position is divisible by 8
            max_lr_x = lr_w - lr_crop_size
            max_lr_y = lr_h - lr_crop_size
            
            lr_x = random.randint(0, max_lr_x // 8) * 8
            lr_y = random.randint(0, max_lr_y // 8) * 8
        else:
            lr_x = ((lr_w - lr_crop_size) // 2 // 8) * 8
            lr_y = ((lr_h - lr_crop_size) // 2 // 8) * 8
        
        hr_x = lr_x * self.scale_factor
        hr_y = lr_y * self.scale_factor
        
        # Crop patches
        lr_patch = lr_img.crop((lr_x, lr_y, lr_x + lr_crop_size, lr_y + lr_crop_size))
        hr_patch = hr_img.crop((hr_x, hr_y, hr_x + self.crop_size, hr_y + self.crop_size))
        
        # Final check - ensure dimensions are correct
        assert lr_patch.size[0] == lr_crop_size and lr_patch.size[1] == lr_crop_size
        assert hr_patch.size[0] == self.crop_size and hr_patch.size[1] == self.crop_size
        
        return hr_patch, lr_patch
    
    def _to_tensor(self, img):
        """Converts PIL Image to tensor in [0, 1]."""
        img_array = np.array(img).astype(np.float32) / 255.0
        
        if len(img_array.shape) == 3:
            img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1))
        else:
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
        
        return img_tensor
    
    def __getitem__(self, idx):
        # Map idx to actual pair index
        real_idx = idx % len(self.image_pairs)
        hr_path, lr_path = self.image_pairs[real_idx]
        
        try:
            # Load images
            hr_img = Image.open(hr_path).convert('RGB')
            lr_img = Image.open(lr_path).convert('RGB')
            
            # Random/center crop
            hr_patch, lr_patch = self._random_crop(hr_img, lr_img)
            
            # Augmentations for train only
            if self.mode == 'train' and random.random() > 0.5:
                # Horizontal flip
                hr_patch = hr_patch.transpose(Image.FLIP_LEFT_RIGHT)
                lr_patch = lr_patch.transpose(Image.FLIP_LEFT_RIGHT)
            
            if self.mode == 'train' and random.random() > 0.5:
                # Vertical flip
                hr_patch = hr_patch.transpose(Image.FLIP_TOP_BOTTOM)
                lr_patch = lr_patch.transpose(Image.FLIP_TOP_BOTTOM)
            
            # Convert to tensors
            hr_tensor = self._to_tensor(hr_patch)
            lr_tensor = self._to_tensor(lr_patch)
            
            return {
                'lr': lr_tensor,
                'hr': hr_tensor,
                'hr_path': hr_path,
                'lr_path': lr_path
            }
            
        except Exception as e:
            print(f"❌ Loading error for {hr_path}, {lr_path}: {e}")
            # Fallback to first sample
            if real_idx > 0:
                return self.__getitem__(0)
            else:
                # Return dummy data
                dummy_lr = torch.zeros(3, 64, 64)   # x2 => 64x64 LR
                dummy_hr = torch.zeros(3, 128, 128) # x2 => 128x128 HR
                return {'lr': dummy_lr, 'hr': dummy_hr, 'hr_path': '', 'lr_path': ''}

class Swin2SRFinetuner:
    """Class for fine-tuning Swin2SR x2."""
    
    def __init__(self, model_name="caidas/swin2SR-lightweight-x2-64", device='cuda'):
        self.model_name = model_name
        self.device = device
        
        print(f"🔧 Loading {model_name}...")
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 CUDA cache cleared")
            
            # Check available memory
            device_idx = int(device.split(':')[1]) if ':' in device else 0
            total_memory = torch.cuda.get_device_properties(device_idx).total_memory / 1e9
            allocated_memory = torch.cuda.memory_allocated(device_idx) / 1e9
            available_memory = total_memory - allocated_memory
            
            print(f"💾 GPU Memory - Total: {total_memory:.1f}GB, Available: {available_memory:.1f}GB")
            
            if available_memory < 4.0:
                print("⚠️ Low GPU memory. Try reducing batch_size or crop_size")
        
        # Load processor
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        
        # FIX: Always load model in float32 for stability
        self.model = Swin2SRForImageSuperResolution.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        # Move model to device
        self.model.to(device)
        
        # Parameter summary
        param_count = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"✅ Model loaded on {device}")
        print(f"📊 Total parameters: {param_count/1e6:.1f}M")
        print(f"🎯 Trainable parameters: {trainable_params/1e6:.1f}M")
        print(f"🔢 Model dtype: {next(self.model.parameters()).dtype}")
        print("🎯 Strategy: Single x2 fine-tuning")
    
    def prepare_batch(self, batch):
        """Prepares a batch for training."""
        lr_images = batch['lr'].to(self.device)
        hr_images = batch['hr'].to(self.device)
        
        # Convert LR tensors to format expected by processor
        lr_pil_images = []
        for i in range(lr_images.shape[0]):
            lr_tensor = lr_images[i]
            lr_array = (lr_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            lr_pil = Image.fromarray(lr_array)
            lr_pil_images.append(lr_pil)
        
        # Process with image processor
        inputs = self.processor(lr_pil_images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs, hr_images
    
    def compute_loss(self, outputs, hr_targets):
        """Computes loss function with size matching."""
        pred_hr = outputs.reconstruction
        
        # FIX: Match output size to target size when needed
        if pred_hr.shape != hr_targets.shape:
            target_h, target_w = hr_targets.shape[-2:]
            pred_hr = F.interpolate(
                pred_hr, 
                size=(target_h, target_w), 
                mode='bilinear', 
                align_corners=False
            )
        
        # Main loss: L1
        l1_loss = F.l1_loss(pred_hr, hr_targets)
        
        # Secondary loss: MSE
        mse_loss = F.mse_loss(pred_hr, hr_targets)
        
        # Total loss
        total_loss = l1_loss + 0.1 * mse_loss
        
        return {
            'total_loss': total_loss,
            'l1_loss': l1_loss,
            'mse_loss': mse_loss
        }
    
    def train_epoch(self, dataloader, optimizer, scheduler=None, use_amp=False):
        """Trains one epoch."""
        self.model.train()
        
        total_loss = 0
        total_l1 = 0
        total_mse = 0
        
        pbar = tqdm(dataloader, desc="Training (x2)")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # Prepare batch
                inputs, hr_targets = self.prepare_batch(batch)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(**inputs)
                
                # Compute loss
                losses = self.compute_loss(outputs, hr_targets)
                
                # Backward pass
                losses['total_loss'].backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Update metrics
                total_loss += losses['total_loss'].item()
                total_l1 += losses['l1_loss'].item()
                total_mse += losses['mse_loss'].item()
                
                # Update progress bar
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'L1': f'{total_l1/(batch_idx+1):.4f}',
                    'MSE': f'{total_mse/(batch_idx+1):.4f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })
                
                # Clear cache every few batches
                if (batch_idx + 1) % 50 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"❌ Error in batch {batch_idx}: {e}")
                torch.cuda.empty_cache()
                continue
        
        if scheduler:
            scheduler.step()
        
        return {
            'avg_loss': total_loss / len(dataloader),
            'avg_l1': total_l1 / len(dataloader),
            'avg_mse': total_mse / len(dataloader)
        }
    
    def validate(self, dataloader):
        """Validation."""
        self.model.eval()
        
        total_loss = 0
        total_l1 = 0
        total_mse = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validation (x2)")):
                try:
                    inputs, hr_targets = self.prepare_batch(batch)
                    outputs = self.model(**inputs)
                    losses = self.compute_loss(outputs, hr_targets)
                    
                    total_loss += losses['total_loss'].item()
                    total_l1 += losses['l1_loss'].item()
                    total_mse += losses['mse_loss'].item()
                    
                    # Clear cache every few batches
                    if (batch_idx + 1) % 25 == 0:
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"❌ Validation batch error {batch_idx}: {e}")
                    torch.cuda.empty_cache()
                    continue
        
        return {
            'avg_loss': total_loss / len(dataloader),
            'avg_l1': total_l1 / len(dataloader),
            'avg_mse': total_mse / len(dataloader)
        }
    
    def save_model(self, output_dir, epoch, best=False):
        """Saves model and processor."""
        os.makedirs(output_dir, exist_ok=True)
        
        if best:
            model_path = os.path.join(output_dir, 'best_model')
        else:
            model_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}')
        
        # Save model and processor
        self.model.save_pretrained(model_path)
        self.processor.save_pretrained(model_path)
        
        print(f"💾 Model saved: {model_path}")
        return model_path

def main():
    """Main fine-tuning function for x2."""
    
    print("🚀 SWIN2SR FINE-TUNING - x2 SINGLE PASS")
    print("🎯 Strategy: LR_2x → Single Swin2SR x2 → HR")
    print("=" * 60)
    
    # Configuration with early stopping for x2
    config = {
        'model_name': 'caidas/swin2SR-lightweight-x2-64',
        # PATHS FOR x2:
        'train_hr_dir': 'swin2_wr/train/HR',
        'train_lr_dir': 'swin2_wr/train/LR_2x',    # Train on LR_2x
        'val_hr_dir': 'swin2_wr/val/HR', 
        'val_lr_dir': 'swin2_wr/val/LR_2x',        # Validate on LR_2x
        'output_dir': 'fine_tuned_swin2sr_2x',
        'crop_size': 128,        # HR crop size (LR will be 64x64)
        'scale_factor': 2,       # Target scale x2
        'batch_size': 32,        # Larger batch for x2 (less memory)
        'epochs': 100,
        'learning_rate': 1e-5,
        'device': 'cuda:1' if torch.cuda.is_available() else 'cpu',
        # EARLY STOPPING CONFIG:
        'early_stopping_patience': 15,
        'min_delta': 1e-4,
        'val_frequency': 5
    }
    
    print("📋 x2 CONFIGURATION (SINGLE PASS):")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    print("\n🔧 x2 STRATEGY:")
    print("   📁 Input: LR_2x (64x64 crops)")
    print("   🎯 Target: HR (128x128 crops)")  
    print("   🔄 Method: LR → Single Swin2SR x2 → HR")
    print("   💾 Batch size: 32 (standard)")
    print("   🧠 Learning rate: 1e-5 (standard)")
    print(f"   🛑 Early stopping: {config['early_stopping_patience']} checks patience")
    
    # Check required folders
    required_dirs = [
        config['train_hr_dir'], config['train_lr_dir'],
        config['val_hr_dir'], config['val_lr_dir']
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"❌ Missing folder: {dir_path}")
            print("💡 Set correct paths in config")
            return
    
    # Create datasets
    print("\n📁 Creating datasets...")
    train_dataset = SRDataset(
        config['train_hr_dir'],
        config['train_lr_dir'],
        config['crop_size'],
        config['scale_factor'],
        mode='train'
    )
    
    val_dataset = SRDataset(
        config['val_hr_dir'],
        config['val_lr_dir'],
        config['crop_size'],
        config['scale_factor'],
        mode='val'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"✅ Train batches: {len(train_loader)}")
    print(f"✅ Val batches: {len(val_loader)}")
    
    # Clear memory before loading model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Initialize finetuner
    print("\n🔧 Initializing model for x2...")
    try:
        finetuner = Swin2SRFinetuner(config['model_name'], config['device'])
    except torch.cuda.OutOfMemoryError:
        print("❌ CUDA Out of Memory while loading model")
        print("💡 Try:")
        print("   1. Reduce crop_size to 64")
        print("   2. Reduce batch_size to 16 or 8")
        print("   3. Use device='cpu'")
        return
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return
    
    # Optimizer for x2
    optimizer = optim.AdamW(
        finetuner.model.parameters(),
        lr=config['learning_rate'],
        weight_decay=1e-4,
        eps=1e-8
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=1e-7
    )
    
    # Training history
    train_history = []
    val_history = []
    best_val_loss = float('inf')
    
    # Early stopping variables
    epochs_without_improvement = 0
    early_stop = False
    
    print("\n🚀 Starting x2 fine-tuning...")
    print("=" * 60)
    
    # Training loop with early stopping
    for epoch in range(config['epochs']):
        if early_stop:
            print(f"\n🛑 EARLY STOPPING after {epoch} epochs")
            print(f"📊 No improvement for {config['early_stopping_patience']} validation checks")
            break
            
        print(f"\n📅 EPOCH {epoch+1}/{config['epochs']}")
        print("-" * 50)
        
        # Clear memory at start of epoch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            # Training with single x2 pass
            train_metrics = finetuner.train_epoch(
                train_loader, optimizer, scheduler, use_amp=False
            )
            train_history.append(train_metrics)
            
            print(f"🎯 Train - Loss: {train_metrics['avg_loss']:.4f}, "
                  f"L1: {train_metrics['avg_l1']:.4f}, MSE: {train_metrics['avg_mse']:.4f}")
            
            # Validation every val_frequency epochs
            if (epoch + 1) % config['val_frequency'] == 0:
                val_metrics = finetuner.validate(val_loader)
                val_history.append(val_metrics)
                
                current_val_loss = val_metrics['avg_loss']
                
                print(f"✅ Val - Loss: {current_val_loss:.4f}, "
                      f"L1: {val_metrics['avg_l1']:.4f}, MSE: {val_metrics['avg_mse']:.4f}")
                
                # Early stopping logic
                if current_val_loss < (best_val_loss - config['min_delta']):
                    # Significant improvement
                    improvement = best_val_loss - current_val_loss
                    best_val_loss = current_val_loss
                    epochs_without_improvement = 0
                    
                    finetuner.save_model(config['output_dir'], epoch, best=True)
                    print(f"🏆 New best x2 model! Val Loss: {best_val_loss:.4f} (↓{improvement:.4f})")
                    
                else:
                    # No significant improvement
                    epochs_without_improvement += 1
                    print(f"⏳ No improvement: {epochs_without_improvement}/{config['early_stopping_patience']}")
                    
                    if epochs_without_improvement >= config['early_stopping_patience']:
                        early_stop = True
                        print("🛑 Early stopping triggered")
            
            # Check GPU memory
            if torch.cuda.is_available() and (epoch + 1) % 10 == 0:
                device_idx = int(config['device'].split(':')[1]) if ':' in config['device'] else 0
                allocated = torch.cuda.memory_allocated(device_idx) / 1e9
                print(f"💾 GPU Memory allocated: {allocated:.1f}GB")
        
        except torch.cuda.OutOfMemoryError:
            print(f"❌ CUDA OOM in epoch {epoch+1}")
            print("🔧 Trying to clear memory...")
            gc.collect()
            torch.cuda.empty_cache()
            continue
        
        except Exception as e:
            print(f"❌ Error in epoch {epoch+1}: {e}")
            continue
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            finetuner.save_model(config['output_dir'], epoch)
    
    # Summary
    print("\n🎉 x2 FINE-TUNING COMPLETED")
    print("=" * 60)
    if early_stop:
        print(f"🛑 Stopped by early stopping after {epoch+1} epochs")
        print(f"⏰ Last improvement: {config['early_stopping_patience'] * config['val_frequency']} epochs ago")
    else:
        print(f"✅ Full training completed ({config['epochs']} epochs)")
    
    print(f"🏆 Best x2 model: {config['output_dir']}/best_model")
    print(f"📊 Best val loss: {best_val_loss:.4f}")
    print(f"📁 All checkpoints: {config['output_dir']}")
    
    # Plot results with early stopping marker
    if train_history:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot([h['avg_loss'] for h in train_history], label='Train Loss')
        if val_history:
            val_epochs = [i * config['val_frequency'] for i in range(len(val_history))]
            plt.plot(val_epochs, [h['avg_loss'] for h in val_history], label='Val Loss')
        
        # Mark early stopping point
        if early_stop:
            plt.axvline(x=epoch+1, color='red', linestyle='--', alpha=0.7, label='Early Stop')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Loss x2 (Single Pass Strategy)')
        
        plt.subplot(1, 2, 2)
        plt.plot([h['avg_l1'] for h in train_history], label='Train L1')
        if val_history:
            plt.plot(val_epochs, [h['avg_l1'] for h in val_history], label='Val L1')
        
        if early_stop:
            plt.axvline(x=epoch+1, color='red', linestyle='--', alpha=0.7, label='Early Stop')
            
        plt.xlabel('Epoch')
        plt.ylabel('L1 Loss')
        plt.legend()
        plt.title('L1 Loss x2 (Single Pass Strategy)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(config['output_dir'], 'training_curves_x2.png'))
        print(f"📊 Plots saved: {config['output_dir']}/training_curves_x2.png")

if __name__ == "__main__":
    main()