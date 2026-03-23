import os
import glob
import shutil
import random
from pathlib import Path
import yaml
from collections import defaultdict
from PIL import Image
import numpy as np
import cv2
from datetime import datetime
import torch


def setup_swin2sr_environment():
    """
    Configures the environment for swin2_sr preprocessing.

    Returns:
        dict: Environment information
    """
    print("🔧 Configuring environment for swin2_sr preprocessing...")

    # Check CUDA
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"✅ GPU: {gpu_name}")
        print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
    else:
        device = "cpu"
        print("⚠️ Using CPU")

    return {
        "device": device,
        "cuda_available": cuda_available,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }


def analyze_yolo_dataset_resolutions(yolo_dataset_path):
    """
    Analyzes resolution distribution in the YOLO dataset.

    Args:
        yolo_dataset_path (str): Path to YOLO dataset

    Returns:
        dict: Resolution statistics
    """
    print("📊 Analyzing image resolutions in YOLO dataset...")

    splits = ["train", "val", "test"]
    resolution_stats = defaultdict(lambda: defaultdict(int))
    all_resolutions = []

    for split in splits:
        images_dir = Path(yolo_dataset_path) / "images" / split

        if not images_dir.exists():
            print(f"⚠️ Missing folder: {images_dir}")
            continue

        # Find all images
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
            image_files.extend(glob.glob(str(images_dir / ext)))

        print(f"\n📂 Split {split}: {len(image_files)} images")

        resolutions_in_split = []

        for image_path in image_files:
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    resolution = (width, height)

                    resolutions_in_split.append(resolution)
                    all_resolutions.append(resolution)
                    resolution_stats[split][resolution] += 1

            except Exception as e:
                print(f"❌ Image loading error {image_path}: {e}")

        # Per-split statistics
        unique_resolutions = set(resolutions_in_split)
        print(f"   📏 Unique resolutions: {len(unique_resolutions)}")

        # Top 5 most common resolutions
        resolution_counts = defaultdict(int)
        for res in resolutions_in_split:
            resolution_counts[res] += 1

        top_resolutions = sorted(resolution_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for (w, h), count in top_resolutions:
            print(f"   📐 {w}x{h}: {count} images ({count/len(resolutions_in_split)*100:.1f}%)")

    # Global statistics
    print(f"\n📈 GLOBAL STATISTICS:")
    total_images = len(all_resolutions)
    unique_global = set(all_resolutions)
    print(f"   📸 Total images: {total_images}")
    print(f"   📏 Unique resolutions: {len(unique_global)}")

    # Find most common resolution as a reference
    global_counts = defaultdict(int)
    for res in all_resolutions:
        global_counts[res] += 1

    most_common_resolution = max(global_counts.items(), key=lambda x: x[1])
    print(f"   🎯 Most common resolution: {most_common_resolution[0]} ({most_common_resolution[1]} images)")

    return {
        "resolution_stats": resolution_stats,
        "total_images": total_images,
        "unique_resolutions": len(unique_global),
        "target_resolution": most_common_resolution[0],
        "all_resolutions": all_resolutions,
    }


def resize_image_preserve_resolution(image, scale_factor, method="bicubic"):
    """
    Resizes image while preserving original resolution as the final upscaling target.

    Args:
        image: PIL Image
        scale_factor: Scaling factor (2, 4)
        method: Interpolation method

    Returns:
        PIL Image: Resized image
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    original_width, original_height = image.size

    # Compute downscaled size
    new_width = original_width // scale_factor
    new_height = original_height // scale_factor

    # Ensure dimensions are > 0
    new_width = max(1, new_width)
    new_height = max(1, new_height)

    if method == "bicubic":
        resized = image.resize((new_width, new_height), Image.BICUBIC)
    elif method == "lanczos":
        resized = image.resize((new_width, new_height), Image.LANCZOS)
    else:
        resized = image.resize((new_width, new_height), Image.BILINEAR)

    return resized


def create_lr_hr_pairs_preserve_resolution(image_path, output_dirs):
    """
    Creates LR-HR pairs while preserving original resolution as HR target.

    Args:
        image_path (str): Path to source image
        output_dirs (dict): Output path mapping

    Returns:
        dict: Information about created pairs
    """
    try:
        # Load image
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")

            # HR = original image (unchanged)
            hr_image = img.copy()
            original_width, original_height = hr_image.size

            # Check minimum size for 4x downscaling
            if original_width < 4 or original_height < 4:
                return {
                    "success": False,
                    "error": f"Image too small for 4x downscaling: {original_width}x{original_height}",
                }

            # Create LR variants by downscaling (2x and 4x only)
            lr_2x = resize_image_preserve_resolution(hr_image, scale_factor=2, method="bicubic")
            lr_4x = resize_image_preserve_resolution(hr_image, scale_factor=4, method="bicubic")

            # Generate filenames
            base_name = Path(image_path).stem

            # Save all variants
            hr_path = output_dirs["HR"] / f"{base_name}.png"
            lr_2x_path = output_dirs["LR_2x"] / f"{base_name}.png"
            lr_4x_path = output_dirs["LR_4x"] / f"{base_name}.png"

            hr_image.save(hr_path, "PNG")
            lr_2x.save(lr_2x_path, "PNG")
            lr_4x.save(lr_4x_path, "PNG")

            return {
                "success": True,
                "original_resolution": (original_width, original_height),
                "hr_size": (original_width, original_height),
                "lr_2x_size": lr_2x.size,
                "lr_4x_size": lr_4x.size,
                "files": {"HR": hr_path, "LR_2x": lr_2x_path, "LR_4x": lr_4x_path},
            }

    except Exception as e:
        return {"success": False, "error": str(e)}


def setup_swin2sr_directory_structure(output_dir):
    """
    Creates folder structure for a swin2_sr dataset.

    Args:
        output_dir (str): Main output directory

    Returns:
        tuple: (dataset_root, dir_structure)
    """
    print("📁 Creating swin2_sr dataset folder structure...")

    dataset_root = Path(output_dir)

    # Structure: split/resolution_type/
    splits = ["train", "val", "test"]
    res_types = ["HR", "LR_2x", "LR_4x"]

    dir_structure = {}

    for split in splits:
        dir_structure[split] = {}
        for res_type in res_types:
            dir_path = dataset_root / split / res_type
            dir_path.mkdir(parents=True, exist_ok=True)
            dir_structure[split][res_type] = dir_path
            print(f"✅ Created: {dir_path}")

    return dataset_root, dir_structure


def process_yolo_dataset_for_swin2sr_preserve_resolution(yolo_dataset_path, swin2sr_output_dir):
    """
    Processes YOLO dataset for swin2_sr training while preserving original resolutions.

    Args:
        yolo_dataset_path (str): Path to YOLO dataset
        swin2sr_output_dir (str): swin2_sr output folder

    Returns:
        dict: Processing statistics
    """
    print("🔄 Processing YOLO dataset for swin2_sr (preserving original resolutions)...")

    # Create folder structure
    dataset_root, dir_structure = setup_swin2sr_directory_structure(swin2sr_output_dir)

    processing_stats = defaultdict(lambda: defaultdict(int))
    failed_files = []
    resolution_stats = defaultdict(list)

    splits = ["train", "val", "test"]

    for split in splits:
        print(f"\n📂 Processing split: {split}")

        yolo_images_dir = Path(yolo_dataset_path) / "images" / split
        if not yolo_images_dir.exists():
            print(f"⚠️ Missing folder: {yolo_images_dir}")
            continue

        # Find all images
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
            image_files.extend(glob.glob(str(yolo_images_dir / ext)))

        print(f"   📸 Found {len(image_files)} images")

        for i, image_path in enumerate(image_files):
            try:
                result = create_lr_hr_pairs_preserve_resolution(image_path, dir_structure[split])

                if result["success"]:
                    processing_stats[split]["success"] += 1
                    resolution_stats[split].append(result["original_resolution"])

                    # Log first examples
                    if i < 10:
                        orig_res = result["original_resolution"]
                        lr_2x_res = result["lr_2x_size"]
                        lr_4x_res = result["lr_4x_size"]
                        print(f"   📐 {Path(image_path).name}: {orig_res} → 2x:{lr_2x_res} 4x:{lr_4x_res}")
                else:
                    processing_stats[split]["failed"] += 1
                    failed_files.append((image_path, result["error"]))

                if (i + 1) % 100 == 0:
                    print(f"   📋 Processed {i + 1}/{len(image_files)} images...")

            except Exception as e:
                processing_stats[split]["failed"] += 1
                failed_files.append((image_path, str(e)))
                print(f"❌ Processing error {image_path}: {e}")

        print(f"   ✅ Success: {processing_stats[split]['success']}")
        print(f"   ❌ Failed: {processing_stats[split]['failed']}")

        if resolution_stats[split]:
            unique_resolutions = set(resolution_stats[split])
            print(f"   📏 Unique resolutions in {split}: {len(unique_resolutions)}")

    return {
        "swin2_sr_root": dataset_root,
        "processing_stats": processing_stats,
        "failed_files": failed_files,
        "resolution_stats": resolution_stats,
        "preserve_original": True,
    }


def calculate_baseline_metrics_variable_resolution(swin2_sr_dataset_path):
    """
    Calculates baseline PSNR/SSIM metrics for variable resolution at different scales.

    Args:
        swin2_sr_dataset_path (str): Path to swin2_sr dataset

    Returns:
        dict: Baseline metrics
    """
    print("📊 Calculating baseline metrics (PSNR/SSIM) for variable resolution...")

    def calculate_psnr(img1, img2):
        """Calculates PSNR between two images."""
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float("inf")
        return 20 * np.log10(255.0 / np.sqrt(mse))

    def calculate_ssim(img1, img2):
        """Calculates SSIM between two images."""
        try:
            from skimage.metrics import structural_similarity
            return structural_similarity(img1, img2, multichannel=True, channel_axis=2)
        except Exception:
            return 0.0

    dataset_root = Path(swin2_sr_dataset_path)

    # Validation set test (2x and 4x only)
    hr_dir = dataset_root / "val" / "HR"
    lr_2x_dir = dataset_root / "val" / "LR_2x"
    lr_4x_dir = dataset_root / "val" / "LR_4x"

    if not all(d.exists() for d in [hr_dir, lr_2x_dir, lr_4x_dir]):
        print("⚠️ Validation folders not found (HR, LR_2x, LR_4x)")
        return {}

    hr_files = set(f.stem for f in hr_dir.glob("*.png"))
    lr_2x_files = set(f.stem for f in lr_2x_dir.glob("*.png"))
    lr_4x_files = set(f.stem for f in lr_4x_dir.glob("*.png"))

    common_files = list(hr_files & lr_2x_files & lr_4x_files)[:50]  # Sample 50 for speed
    print(f"   📊 Testing on {len(common_files)} images...")

    metrics = defaultdict(list)
    resolution_examples = []

    for file_stem in common_files:
        try:
            hr_path = hr_dir / f"{file_stem}.png"
            lr_2x_path = lr_2x_dir / f"{file_stem}.png"
            lr_4x_path = lr_4x_dir / f"{file_stem}.png"

            hr_img = np.array(Image.open(hr_path))
            lr_2x_img = np.array(Image.open(lr_2x_path))
            lr_4x_img = np.array(Image.open(lr_4x_path))

            hr_resolution = hr_img.shape[:2][::-1]  # (width, height)

            # Bicubic baseline upscaling to HR size
            lr_2x_upscaled = np.array(Image.fromarray(lr_2x_img).resize(hr_resolution, Image.BICUBIC))
            lr_4x_upscaled = np.array(Image.fromarray(lr_4x_img).resize(hr_resolution, Image.BICUBIC))

            psnr_2x = calculate_psnr(hr_img, lr_2x_upscaled)
            psnr_4x = calculate_psnr(hr_img, lr_4x_upscaled)

            ssim_2x = calculate_ssim(hr_img, lr_2x_upscaled)
            ssim_4x = calculate_ssim(hr_img, lr_4x_upscaled)

            metrics["psnr_2x"].append(psnr_2x)
            metrics["psnr_4x"].append(psnr_4x)
            metrics["ssim_2x"].append(ssim_2x)
            metrics["ssim_4x"].append(ssim_4x)

            if len(resolution_examples) < 5:
                resolution_examples.append(
                    {
                        "file": file_stem,
                        "hr": hr_resolution,
                        "lr_2x": lr_2x_img.shape[:2][::-1],
                        "lr_4x": lr_4x_img.shape[:2][::-1],
                    }
                )

        except Exception as e:
            print(f"❌ Metrics calculation error for {file_stem}: {e}")

    avg_metrics = {}
    for key, values in metrics.items():
        if values:
            avg_metrics[key] = np.mean(values)

    print(f"\n📈 BASELINE METRICS (bicubic upscaling, variable resolution):")
    print(f"   🎯 2x upscaling: PSNR={avg_metrics.get('psnr_2x', 0):.2f}dB, SSIM={avg_metrics.get('ssim_2x', 0):.4f}")
    print(f"   🎯 4x upscaling: PSNR={avg_metrics.get('psnr_4x', 0):.2f}dB, SSIM={avg_metrics.get('ssim_4x', 0):.4f}")

    print(f"\n📐 Resolution examples:")
    for ex in resolution_examples:
        print(f"   {ex['file']}: HR={ex['hr']}, LR_2x={ex['lr_2x']}, LR_4x={ex['lr_4x']}")

    avg_metrics["resolution_examples"] = resolution_examples
    return avg_metrics


def create_swin2sr_config_yaml_variable_resolution(swin2_sr_dataset_path, baseline_metrics, processing_result):
    """
    Creates configuration file for swin2_sr training with variable resolution.
    """
    print("📄 Creating swin2_sr configuration (variable resolution)...")

    dataset_root = Path(swin2_sr_dataset_path)

    # Count files in each split
    splits_info = {}
    for split in ["train", "val", "test"]:
        hr_dir = dataset_root / split / "HR"
        splits_info[split] = len(list(hr_dir.glob("*.png"))) if hr_dir.exists() else 0

    # Resolution analysis
    resolution_analysis = {}
    for split, resolutions in processing_result["resolution_stats"].items():
        if resolutions:
            unique_res = set(resolutions)
            resolution_analysis[split] = {
                "unique_count": len(unique_res),
                "total_images": len(resolutions),
                "resolution_range": {
                    "min_width": min(r[0] for r in resolutions),
                    "max_width": max(r[0] for r in resolutions),
                    "min_height": min(r[1] for r in resolutions),
                    "max_height": max(r[1] for r in resolutions),
                },
            }

    # swin2_sr configuration for scales [2, 4]
    swin2sr_config = {
        "dataset_info": {
            "name": "ExDark_swin2_sr_Variable_Resolution",
            "description": "ExDark dataset for swin2_sr with preserved original resolutions",
            "resolution_strategy": "preserve_original",
            "scales": [2, 4],
            "total_images": sum(splits_info.values()),
        },
        "resolution_analysis": resolution_analysis,
        "splits": {
            "train": {
                "images": splits_info["train"],
                "hr_path": str(dataset_root / "train" / "HR"),
                "lr_2x_path": str(dataset_root / "train" / "LR_2x"),
                "lr_4x_path": str(dataset_root / "train" / "LR_4x"),
            },
            "val": {
                "images": splits_info["val"],
                "hr_path": str(dataset_root / "val" / "HR"),
                "lr_2x_path": str(dataset_root / "val" / "LR_2x"),
                "lr_4x_path": str(dataset_root / "val" / "LR_4x"),
            },
            "test": {
                "images": splits_info["test"],
                "hr_path": str(dataset_root / "test" / "HR"),
                "lr_2x_path": str(dataset_root / "test" / "LR_2x"),
                "lr_4x_path": str(dataset_root / "test" / "LR_4x"),
            },
        },
        "baseline_metrics": baseline_metrics,
        "training_strategy": {
            "model": "swin2_sr",
            "scale": 2,
            "resolution_handling": "dynamic_batching_or_patch_based",
            "cascade_strategy": {
                "2x": "1x swin2_sr_x2",
                "4x": "2x swin2_sr_x2 (cascade)",
            },
            "expected_improvement": {
                "psnr_gain": "3-5 dB over bicubic",
                "ssim_gain": "0.05-0.15 over bicubic",
            },
        },
        "implementation_notes": {
            "batch_handling": "Use patch-based training or dynamic batching for variable resolutions",
            "coordinate_mapping": "Perfect 1:1 mapping - no coordinate transformation needed",
            "yolo_integration": "Direct bbox mapping without coordinate conversion",
        },
        "a100_recommendations": {
            "batch_size": "4-8 (dynamic batching required for variable resolution)",
            "estimated_time": "3-5 days on A100 40GB",
            "memory_usage": "15-30GB depending on largest images",
        },
    }

    config_path = dataset_root / "swin2_sr_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(swin2sr_config, f, default_flow_style=False, sort_keys=False)

    print(f"✅ Configuration saved: {config_path}")
    print(f"\n📄 swin2_sr_config.yaml summary:")
    print(f"   📊 Dataset: {swin2sr_config['dataset_info']['name']}")
    print(f"   🎯 Resolution strategy: {swin2sr_config['dataset_info']['resolution_strategy']}")
    print(f"   📸 Total images: {swin2sr_config['dataset_info']['total_images']}")
    print(f"   🔄 Scales: {swin2sr_config['dataset_info']['scales']}")
    print(f"   🚀 Strategy: swin2_sr 2x with cascade, variable resolution")


def save_processing_report(swin2_sr_dataset_path, processing_result, env_info):
    """
    Saves a detailed processing report.

    Args:
        swin2_sr_dataset_path (str): Path to swin2_sr dataset
        processing_result (dict): Processing results
        env_info (dict): Environment information
    """
    print("📊 Saving processing report...")

    dataset_root = Path(swin2_sr_dataset_path)
    report_path = dataset_root / "processing_report.txt"

    with open(report_path, "w") as f:
        f.write("YOLO → SWIN2_SR DATASET PROCESSING REPORT\n")
        f.write("=" * 50 + "\n\n")

        f.write("PROCESSING INFO:\n")
        f.write("-" * 18 + "\n")
        f.write(f"Timestamp: {env_info['timestamp']}\n")
        f.write("Source: YOLO ExDark dataset\n")
        f.write("Target: swin2_sr super-resolution dataset\n")
        f.write("Resolution strategy: Preserve original per image\n")
        f.write("Scales: 2x, 4x only\n\n")

        f.write("PROCESSING STATISTICS:\n")
        f.write("-" * 25 + "\n")

        total_success = 0
        total_failed = 0

        for split, stats in processing_result["processing_stats"].items():
            success = stats["success"]
            failed = stats["failed"]
            total = success + failed

            total_success += success
            total_failed += failed

            f.write(f"{split.upper()}:\n")
            f.write(f"   Success: {success}/{total} ({success/total*100:.1f}%)\n")
            f.write(f"   Failed: {failed}/{total} ({failed/total*100:.1f}%)\n\n")

        f.write("TOTAL:\n")
        f.write(f"   Success: {total_success}\n")
        f.write(f"   Failed: {total_failed}\n")
        f.write(f"   Success rate: {total_success/(total_success+total_failed)*100:.1f}%\n\n")

        f.write("RESOLUTION ANALYSIS:\n")
        f.write("-" * 21 + "\n")
        for split, resolutions in processing_result["resolution_stats"].items():
            if resolutions:
                unique_res = set(resolutions)
                f.write(f"{split.upper()}:\n")
                f.write(f"   Unique resolutions: {len(unique_res)}\n")
                f.write(f"   Total images: {len(resolutions)}\n")
                f.write(f"   Width range: {min(r[0] for r in resolutions)} - {max(r[0] for r in resolutions)}\n")
                f.write(f"   Height range: {min(r[1] for r in resolutions)} - {max(r[1] for r in resolutions)}\n\n")

        if processing_result["failed_files"]:
            f.write("FAILED FILES:\n")
            f.write("-" * 15 + "\n")
            for file_path, error in processing_result["failed_files"]:
                f.write(f"   {file_path}: {error}\n")
            f.write("\n")

        f.write("DATASET STRUCTURE:\n")
        f.write("-" * 20 + "\n")
        f.write("swin2_sr_dataset/\n")
        f.write("├── train/\n")
        f.write("│   ├── HR/      # Original resolution targets\n")
        f.write("│   ├── LR_2x/   # Original/2 resolution inputs\n")
        f.write("│   └── LR_4x/   # Original/4 resolution inputs\n")
        f.write("├── val/\n")
        f.write("│   └── [same structure as train]\n")
        f.write("├── test/\n")
        f.write("│   └── [same structure as train]\n")
        f.write("├── swin2_sr_config.yaml\n")
        f.write("└── processing_report.txt\n\n")

        f.write("NEXT STEPS:\n")
        f.write("-" * 12 + "\n")
        f.write("1. Review swin2_sr_config.yaml for training parameters\n")
        f.write("2. Train swin2_sr x2 model with dynamic/patch-based batching\n")
        f.write("3. Use cascade approach for 4x upscaling (2x → 2x)\n")
        f.write("4. Evaluate with PSNR/SSIM metrics\n")
        f.write("5. Test on ExDark detection pipeline (direct bbox mapping)\n")
        f.write("6. No coordinate transformation needed!\n")

    print(f"✅ Report saved: {report_path}")


def main():
    """Main preprocessing function: YOLO → swin2_sr with preserved original resolutions."""
    print("🚀 YOLO → SWIN2_SR DATASET PREPROCESSING (PRESERVE ORIGINAL RESOLUTIONS)")
    print("🎯 Preparing ExDark dataset for swin2_sr super-resolution")
    print("🔄 Preserving original resolutions as HR targets")
    print("📐 Scales: 2x and 4x")
    print("=" * 70)

    # Path configuration
    yolo_dataset_path = "yolo_dataset"
    swin2sr_output_dir = "swin2_sr_dataset"

    env_info = setup_swin2sr_environment()

    print(f"\n📋 CONFIGURATION:")
    print(f"   📂 YOLO dataset: {yolo_dataset_path}")
    print(f"   🎯 swin2_sr output: {swin2sr_output_dir}")
    print(f"   📐 Resolution strategy: Preserve original per image")
    print(f"   🔄 Scales: 2x, 4x")

    if not os.path.exists(yolo_dataset_path):
        print(f"❌ Missing YOLO dataset: {yolo_dataset_path}")
        return

    print(f"\n📊 RESOLUTION ANALYSIS:")
    resolution_stats = analyze_yolo_dataset_resolutions(yolo_dataset_path)

    print(f"\n🚨 READY TO PROCESS!")
    print("📐 Each image will keep its original resolution")
    print("🔄 LR variants: HR/2 and HR/4 generated per image")
    print(f"📸 Images to process: {resolution_stats['total_images']}")
    print("⏱️ Estimated time: ~30-60 minutes")
    print("💾 only 2x and 4x")

    user_input = input("\n🤔 Start processing? (y/n): ")
    if user_input.lower() != "y":
        print("❌ Processing canceled by user")
        return

    processing_result = process_yolo_dataset_for_swin2sr_preserve_resolution(
        yolo_dataset_path, swin2sr_output_dir
    )

    baseline_metrics = calculate_baseline_metrics_variable_resolution(
        processing_result["swin2_sr_root"]
    )

    create_swin2sr_config_yaml_variable_resolution(
        processing_result["swin2_sr_root"],
        baseline_metrics,
        processing_result,
    )

    save_processing_report(
        processing_result["swin2_sr_root"],
        processing_result,
        env_info,
    )

    print(f"\n🎉 PREPROCESSING COMPLETED!")
    print("=" * 70)
    print(f"✅ swin2_sr dataset created at: {processing_result['swin2_sr_root']}")
    print(f"📊 Configuration: {processing_result['swin2_sr_root']}/swin2_sr_config.yaml")
    print("📐 Resolution strategy: Preserved original per image")
    print("🔄 Scales: 2x, 4x")

    total_success = sum(stats["success"] for stats in processing_result["processing_stats"].values())
    print(f"📸 Processed images: {total_success}")

if __name__ == "__main__":
    main()