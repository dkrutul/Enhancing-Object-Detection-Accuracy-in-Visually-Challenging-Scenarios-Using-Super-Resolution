import os
import yaml
from pathlib import Path
from datetime import datetime
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd

def setup_training_environment():
    """
    Configures the training environment.
    
    Returns:
        dict: Environment information
    """
    print("🔧 Configuring training environment...")
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"✅ GPU: {gpu_name}")
        print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
    else:
        device = 'cpu'
        print("⚠️ Using CPU - training will be slow!")
    
    # Check PyTorch
    pytorch_version = torch.__version__
    print(f"🔥 PyTorch: {pytorch_version}")
    
    # Check Ultralytics
    try:
        from ultralytics import __version__ as ultralytics_version
        print(f"🚀 Ultralytics: {ultralytics_version}")
    except:
        print("⚠️ Could not check Ultralytics version")
    
    return {
        'device': device,
        'cuda_available': cuda_available,
        'pytorch_version': pytorch_version,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }

def validate_paths(dataset_yaml, pretrained_weights, output_dir):
    """
    Validates whether all required paths exist.
    
    Args:
        dataset_yaml (str): Path to dataset.yaml
        pretrained_weights (str): Path to pretrained weights
        output_dir (str): Output directory
        
    Returns:
        bool: True if everything is valid
    """
    print("📋 Validating paths...")
    
    # Check dataset.yaml
    if not os.path.exists(dataset_yaml):
        print(f"❌ Missing dataset.yaml file: {dataset_yaml}")
        return False
    
    # Check pretrained weights
    if not os.path.exists(pretrained_weights):
        print(f"❌ Missing pretrained weights: {pretrained_weights}")
        return False
    
    # Validate dataset.yaml structure
    try:
        with open(dataset_yaml, 'r') as f:
            dataset_config = yaml.safe_load(f)
        
        required_keys = ['path', 'train', 'val', 'nc', 'names']
        for key in required_keys:
            if key not in dataset_config:
                print(f"❌ Missing key '{key}' in dataset.yaml")
                return False
        
        # Validate data paths
        dataset_path = Path(dataset_config['path'])
        train_path = dataset_path / dataset_config['train']
        val_path = dataset_path / dataset_config['val']
        
        if not train_path.exists():
            print(f"❌ Missing train folder: {train_path}")
            return False
            
        if not val_path.exists():
            print(f"❌ Missing val folder: {val_path}")
            return False
        
        print(f"✅ Dataset: {dataset_config['nc']} classes, {len(dataset_config['names'])} names")
        print(f"✅ Train data: {train_path}")
        print(f"✅ Val data: {val_path}")
        
    except Exception as e:
        print(f"❌ Error parsing dataset.yaml: {e}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"✅ Output dir: {output_dir}")
    
    print(f"✅ Pretrained weights: {pretrained_weights}")
    
    return True

def create_training_config(env_info):
    """
    Creates YOLOv9 training configuration.
    
    Args:
        env_info (dict): Environment information
        
    Returns:
        dict: Training configuration
    """
    print("⚙️ Creating training configuration...")
    
    # Base config
    config = {
        # Training parameters
        'epochs': 100,
        'batch': 8 if env_info['cuda_available'] else 4,
        'imgsz': 640,
        'device': env_info['device'],
        
        # Optimization
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # Augmentation
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
        
        # Validation
        'val': True,
        'save_period': 10,
        'patience': 50,
        
        # Logging
        'verbose': True,
        'plots': True,
        'save': True,
        'save_txt': False,
        'save_conf': False,
        'save_crop': False,
        
        # Other
        'cache': False,
        'workers': 8 if env_info['cuda_available'] else 2,
        'project': None,  # Will be set later
        'name': None,     # Will be set later
        'exist_ok': False,
        'pretrained': True,
        'resume': False,
        'fraction': 1.0,
        'profile': False,
    }
    
    # Adjust batch size based on GPU memory
    if env_info['cuda_available']:
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory >= 24:
                config['batch'] = 16
                print(f"🚀 Large GPU ({gpu_memory:.1f}GB) - batch size: 16")
            elif gpu_memory >= 12:
                config['batch'] = 8
                print(f"🚀 Medium GPU ({gpu_memory:.1f}GB) - batch size: 8")
            else:
                config['batch'] = 4
                print(f"⚠️ Small GPU ({gpu_memory:.1f}GB) - batch size: 4")
        except:
            config['batch'] = 16
            print("⚠️ Could not read GPU memory - batch size: 16")
    
    print(f"⚙️ Epochs: {config['epochs']}")
    print(f"⚙️ Batch size: {config['batch']}")
    print(f"⚙️ Image size: {config['imgsz']}")
    print(f"⚙️ Learning rate: {config['lr0']}")
    print(f"⚙️ Optimizer: {config['optimizer']}")
    
    return config

def run_training(model_path, dataset_yaml, training_config, output_dir, run_name):
    """
    Runs YOLOv9 training.
    
    Args:
        model_path (str): Path to pretrained weights
        dataset_yaml (str): Path to dataset config
        training_config (dict): Training config
        output_dir (str): Output directory
        run_name (str): Run name
        
    Returns:
        tuple: (model, results)
    """
    print(f"\n🚀 STARTING YOLOV9 TRAINING")
    print("=" * 60)
    
    try:
        # Load model
        print(f"📥 Loading model from: {model_path}")
        model = YOLO(model_path)
        
        print(f"✅ Model loaded: {model.model_name}")
        
        # Set project and run name
        training_config['project'] = output_dir
        training_config['name'] = run_name
        
        print(f"📁 Output project: {output_dir}")
        print(f"🏷️ Run name: {run_name}")
        print(f"📊 Dataset: {dataset_yaml}")
        
        # Start training
        print(f"\n🔥 TRAINING STARTED!")
        print(f"⏱️ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        results = model.train(
            data=dataset_yaml,
            **training_config
        )
        
        print("=" * 60)
        print(f"🎉 TRAINING COMPLETED!")
        print(f"⏱️ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return model, results
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def save_training_summary(output_dir, run_name, training_config, env_info, dataset_yaml):
    """
    Saves a training summary.
    
    Args:
        output_dir (str): Output directory
        run_name (str): Run name
        training_config (dict): Training config
        env_info (dict): Environment information
        dataset_yaml (str): Dataset path
    """
    print(f"\n📄 Saving training summary...")
    
    run_dir = Path(output_dir) / run_name
    summary_path = run_dir / "training_summary.txt"
    
    try:
        # Read dataset info
        with open(dataset_yaml, 'r') as f:
            dataset_config = yaml.safe_load(f)
        
        with open(summary_path, 'w') as f:
            f.write("YOLOV9 EXDARK FINE-TUNING SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            # Basic info
            f.write("TRAINING INFO:\n")
            f.write("-" * 15 + "\n")
            f.write(f"Timestamp: {env_info['timestamp']}\n")
            f.write(f"Run name: {run_name}\n")
            f.write(f"Dataset: ExDark → YOLO format\n")
            f.write(f"Model: YOLOv9c fine-tuning\n\n")
            
            # Dataset info
            f.write("DATASET INFO:\n")
            f.write("-" * 15 + "\n")
            f.write(f"Classes: {dataset_config['nc']}\n")
            f.write(f"Class names: {', '.join(dataset_config['names'])}\n")
            f.write(f"Dataset path: {dataset_config['path']}\n")
            f.write(f"Train data: {dataset_config['train']}\n")
            f.write(f"Val data: {dataset_config['val']}\n\n")
            
            # Environment info
            f.write("ENVIRONMENT:\n")
            f.write("-" * 15 + "\n")
            f.write(f"Device: {env_info['device']}\n")
            f.write(f"CUDA available: {env_info['cuda_available']}\n")
            f.write(f"PyTorch version: {env_info['pytorch_version']}\n")
            if env_info['cuda_available']:
                try:
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    f.write(f"GPU: {gpu_name}\n")
                    f.write(f"GPU Memory: {gpu_memory:.1f} GB\n")
                except:
                    pass
            f.write("\n")
            
            # Training config
            f.write("TRAINING CONFIG:\n")
            f.write("-" * 18 + "\n")
            for key, value in training_config.items():
                if key not in ['project', 'name']:  # Skip path-specific configs
                    f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Files info
            f.write("OUTPUT FILES:\n")
            f.write("-" * 15 + "\n")
            f.write("📁 Key files in this directory:\n")
            f.write("   • weights/best.pt - Best model weights\n")
            f.write("   • weights/last.pt - Last epoch weights\n")
            f.write("   • results.png - Training curves\n")
            f.write("   • confusion_matrix.png - Confusion matrix\n")
            f.write("   • val_batch*.jpg - Validation predictions\n")
            f.write("   • train_batch*.jpg - Training batch samples\n")
            f.write("   • args.yaml - Full training arguments\n")
            f.write("   • training_summary.txt - This summary\n")
        
        print(f"✅ Summary saved: {summary_path}")
        
    except Exception as e:
        print(f"❌ Error saving summary: {e}")

def analyze_training_results(output_dir, run_name):
    """
    Analyzes training results and creates additional visualizations.
    
    Args:
        output_dir (str): Output directory
        run_name (str): Run name
    """
    print(f"\n📊 Analyzing training results...")
    
    run_dir = Path(output_dir) / run_name
    results_csv = run_dir / "results.csv"
    
    if not results_csv.exists():
        print(f"⚠️ Missing results.csv file: {results_csv}")
        return
    
    try:
        # Load results
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()  # Remove whitespace
        
        print(f"✅ Loaded {len(df)} epochs from results.csv")
        
        # Find best epoch
        if 'metrics/mAP50(B)' in df.columns:
            best_epoch = df['metrics/mAP50(B)'].idxmax()
            best_map50 = df.loc[best_epoch, 'metrics/mAP50(B)']
            print(f"🏆 Best mAP50: {best_map50:.4f} (epoch {best_epoch + 1})")
        
        # Create additional charts
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        if 'train/box_loss' in df.columns:
            axes[0, 0].plot(df.index, df['train/box_loss'], label='Train Box Loss')
            if 'val/box_loss' in df.columns:
                axes[0, 0].plot(df.index, df['val/box_loss'], label='Val Box Loss')
            axes[0, 0].set_title('Box Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # mAP curves
        if 'metrics/mAP50(B)' in df.columns:
            axes[0, 1].plot(df.index, df['metrics/mAP50(B)'], label='mAP50', color='green')
            if 'metrics/mAP50-95(B)' in df.columns:
                axes[0, 1].plot(df.index, df['metrics/mAP50-95(B)'], label='mAP50-95', color='blue')
            axes[0, 1].set_title('mAP Metrics')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('mAP')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Learning rate
        if 'lr/pg0' in df.columns:
            axes[1, 0].plot(df.index, df['lr/pg0'], label='Learning Rate', color='orange')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('LR')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Precision/Recall
        if 'metrics/precision(B)' in df.columns and 'metrics/recall(B)' in df.columns:
            axes[1, 1].plot(df.index, df['metrics/precision(B)'], label='Precision', color='red')
            axes[1, 1].plot(df.index, df['metrics/recall(B)'], label='Recall', color='purple')
            axes[1, 1].set_title('Precision & Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(run_dir / "detailed_training_curves.png", dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Additional charts saved: detailed_training_curves.png")
        
        # Final summary
        print(f"\n📈 FINAL METRICS:")
        if len(df) > 0:
            final_epoch = df.iloc[-1]
            if 'metrics/mAP50(B)' in final_epoch:
                print(f"   🎯 Final mAP50: {final_epoch['metrics/mAP50(B)']:.4f}")
            if 'metrics/precision(B)' in final_epoch:
                print(f"   🎯 Final Precision: {final_epoch['metrics/precision(B)']:.4f}")
            if 'metrics/recall(B)' in final_epoch:
                print(f"   🎯 Final Recall: {final_epoch['metrics/recall(B)']:.4f}")
        
    except Exception as e:
        print(f"❌ Result analysis error: {e}")

def main():
    """Main YOLOv9 fine-tuning function."""
    print("🚀 YOLOV9 EXDARK FINE-TUNING")
    print("🎯 Fine-tuning YOLOv9c on ExDark dataset")
    print("=" * 60)
    
    # Path configuration
    dataset_yaml = "yolo_dataset/dataset.yaml"
    pretrained_weights = "weights/yolov9c.pt"
    output_dir = "fine_tuned_models"
    
    # Check environment
    env_info = setup_training_environment()
    
    # Create unique run name
    run_name = f"exdark_yolov9c_{env_info['timestamp']}"
    
    print(f"\n📋 CONFIGURATION:")
    print(f"   📊 Dataset: {dataset_yaml}")
    print(f"   🏋️ Pretrained: {pretrained_weights}")
    print(f"   📁 Output: {output_dir}")
    print(f"   🏷️ Run name: {run_name}")
    
    # Validate paths
    if not validate_paths(dataset_yaml, pretrained_weights, output_dir):
        print("❌ Path validation failed!")
        return
    
    # Create training config
    training_config = create_training_config(env_info)
    
    # Confirm start
    print(f"\n🚨 READY TO TRAIN!")
    print(f"⏱️ Estimated time: ~{training_config['epochs'] * 2} minutes")
    print(f"💾 Results will be saved in: {output_dir}/{run_name}")
    
    user_input = input("\n🤔 Start training? (y/n): ")
    if user_input.lower() != 'y':
        print("❌ Training canceled by user")
        return
    
    # Run training
    model, results = run_training(
        pretrained_weights, 
        dataset_yaml, 
        training_config, 
        output_dir, 
        run_name
    )
    
    if model is None:
        print("❌ Training failed!")
        return
    
    # Save summary
    save_training_summary(output_dir, run_name, training_config, env_info, dataset_yaml)
    
    # Analyze results
    analyze_training_results(output_dir, run_name)
    
    # Final summary
    final_model_path = Path(output_dir) / run_name / "weights" / "best.pt"
    
    print(f"\n🎉 FINE-TUNING COMPLETED!")
    print(f"=" * 60)
    print(f"✅ Best model: {final_model_path}")
    print(f"📁 All files: {output_dir}/{run_name}/")
    print(f"📊 Training plots: {output_dir}/{run_name}/results.png")
    print(f"📈 Additional analysis: {output_dir}/{run_name}/detailed_training_curves.png")
    print(f"📄 Summary: {output_dir}/{run_name}/training_summary.txt")

if __name__ == "__main__":
    main()