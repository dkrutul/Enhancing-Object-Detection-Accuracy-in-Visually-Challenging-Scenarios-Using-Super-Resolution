import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

def analyze_image_resolutions(dataset_path):
    """
    Analyzes image sizes in the ExDark dataset.
    
    Args:
        dataset_path (str): Path to the main ExDark folder.
        
    Returns:
        dict: Resolution statistics.
    """
    print("📊 EXDARK DATASET RESOLUTION ANALYSIS")
    print(f"Path: {dataset_path}")
    print("=" * 60)
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return None
    
    # ExDark classes
    exdark_classes = [
        'Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat',
        'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table'
    ]
    
    # Statistics
    resolutions = []
    widths = []
    heights = []
    aspect_ratios = []
    resolution_count = defaultdict(int)
    aspect_ratio_count = defaultdict(int)
    
    total_images = 0
    processed_images = 0
    
    print("🔍 Scanning class folders...")
    
    # Iterate through all class folders
    for class_name in exdark_classes:
        class_path = os.path.join(dataset_path, class_name)
        if os.path.exists(class_path):
            # Find all images in the class folder
            class_images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                class_images.extend(glob.glob(os.path.join(class_path, '**', ext), recursive=True))
            
            print(f"📁 {class_name}: {len(class_images)} images")
            total_images += len(class_images)
            
            # Analyze all images in the class
            for img_path in class_images:
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        
                        # Save data
                        resolutions.append((width, height))
                        widths.append(width)
                        heights.append(height)
                        
                        # Compute aspect ratio
                        aspect_ratio = round(width / height, 2)
                        aspect_ratios.append(aspect_ratio)
                        
                        # Count resolutions
                        resolution_key = f"{width}x{height}"
                        resolution_count[resolution_key] += 1
                        
                        # Categorize aspect ratio
                        if 1.25 <= aspect_ratio <= 1.40:
                            aspect_category = "4:3 (~1.33)"
                        elif 1.70 <= aspect_ratio <= 1.85:
                            aspect_category = "16:9 (~1.78)"
                        elif 0.90 <= aspect_ratio <= 1.10:
                            aspect_category = "1:1 (~1.00)"
                        elif aspect_ratio > 2.0:
                            aspect_category = "Ultra-wide (>2.0)"
                        elif aspect_ratio < 0.8:
                            aspect_category = "Portrait (<0.8)"
                        else:
                            aspect_category = "Other"
                        
                        aspect_ratio_count[aspect_category] += 1
                        processed_images += 1
                        
                except Exception as e:
                    print(f"⚠️ Image read error {img_path}: {e}")
    
    print("\n📈 ANALYSIS RESULTS:")
    print(f"🖼️  Total images: {total_images}")
    print(f"✅ Successfully processed: {processed_images}")
    print(f"❌ Read errors: {total_images - processed_images}")
    
    if processed_images == 0:
        print("❌ No data for analysis")
        return None
    
    # Basic statistics
    print("\n📏 RESOLUTION STATISTICS:")
    print(f"   • Width: {min(widths)} - {max(widths)}px (mean: {np.mean(widths):.1f})")
    print(f"   • Height: {min(heights)} - {max(heights)}px (mean: {np.mean(heights):.1f})")
    print(f"   • Aspect ratio: {min(aspect_ratios):.2f} - {max(aspect_ratios):.2f} (mean: {np.mean(aspect_ratios):.2f})")
    
    # Most common resolutions
    print("\n🔝 TOP 10 MOST COMMON RESOLUTIONS:")
    sorted_resolutions = sorted(resolution_count.items(), key=lambda x: x[1], reverse=True)
    for i, (resolution, count) in enumerate(sorted_resolutions[:10]):
        percentage = (count / processed_images) * 100
        print(f"   {i+1:2d}. {resolution}: {count:4d} ({percentage:5.1f}%)")
    
    # Aspect ratio distribution
    print("\n📐 ASPECT RATIO DISTRIBUTION:")
    for category, count in sorted(aspect_ratio_count.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / processed_images) * 100
        print(f"   • {category}: {count:4d} ({percentage:5.1f}%)")
    
    return {
        'resolutions': resolutions,
        'widths': widths,
        'heights': heights,
        'aspect_ratios': aspect_ratios,
        'resolution_count': dict(resolution_count),
        'aspect_ratio_count': dict(aspect_ratio_count),
        'total_images': total_images,
        'processed_images': processed_images
    }

def create_resolution_plots(stats, save_path="exdark_resolution_analysis.png"):
    """
    Creates resolution analysis plots.
    
    Args:
        stats (dict): Statistics from analyze_image_resolutions.
        save_path (str): Path to save plots.
    """
    if not stats:
        print("❌ No data for visualization")
        return
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('ExDark Dataset - Resolution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Width histogram
    axes[0, 0].hist(stats['widths'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Width distribution', fontweight='bold')
    axes[0, 0].set_xlabel('Width [px]')
    axes[0, 0].set_ylabel('Number of images')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(np.mean(stats['widths']), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(stats["widths"]):.1f}px')
    axes[0, 0].legend()
    
    # 2. Height histogram
    axes[0, 1].hist(stats['heights'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_title('Height distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Height [px]')
    axes[0, 1].set_ylabel('Number of images')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(np.mean(stats['heights']), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(stats["heights"]):.1f}px')
    axes[0, 1].legend()
    
    # 3. Resolution scatter plot
    axes[1, 0].scatter(stats['widths'], stats['heights'], alpha=0.6, s=20, color='purple')
    axes[1, 0].set_title('Resolution distribution (width vs height)', fontweight='bold')
    axes[1, 0].set_xlabel('Width [px]')
    axes[1, 0].set_ylabel('Height [px]')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add lines for common aspect ratios
    x_max = max(stats['widths'])
    y_max = max(stats['heights'])
    x_range = np.linspace(0, x_max, 100)
    
    # 4:3 ratio
    y_43 = x_range / (4/3)
    mask_43 = y_43 <= y_max
    axes[1, 0].plot(x_range[mask_43], y_43[mask_43], 'r--', alpha=0.7, label='4:3')
    
    # 16:9 ratio
    y_169 = x_range / (16/9)
    mask_169 = y_169 <= y_max
    axes[1, 0].plot(x_range[mask_169], y_169[mask_169], 'g--', alpha=0.7, label='16:9')
    
    axes[1, 0].legend()
    
    # 4. Aspect ratio pie chart
    aspect_data = stats['aspect_ratio_count']
    labels = list(aspect_data.keys())
    sizes = list(aspect_data.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    
    wedges, texts, autotexts = axes[1, 1].pie(
        sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90
    )
    axes[1, 1].set_title('Image aspect ratio distribution', fontweight='bold')
    
    # Adjust font
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Plots saved as: {save_path}")
    plt.show()

def create_top_resolutions_chart(stats, save_path="exdark_top_resolutions.png", top_n=15):
    """
    Creates a bar chart of the most frequent resolutions.
    
    Args:
        stats (dict): Statistics from analyze_image_resolutions.
        save_path (str): Path to save the chart.
        top_n (int): Number of most frequent resolutions to display.
    """
    if not stats:
        print("❌ No data for visualization")
        return
    
    # Get top N resolutions
    sorted_resolutions = sorted(
        stats['resolution_count'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]
    
    resolutions = [item[0] for item in sorted_resolutions]
    counts = [item[1] for item in sorted_resolutions]
    percentages = [(count / stats['processed_images']) * 100 for count in counts]
    
    # Create chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    bars = ax.bar(range(len(resolutions)), counts, color='steelblue', alpha=0.8)
    
    # Add labels
    ax.set_xlabel('Resolution', fontweight='bold', fontsize=12)
    ax.set_ylabel('Number of images', fontweight='bold', fontsize=12)
    ax.set_title(f'Top {top_n} most common resolutions in ExDark',
                 fontweight='bold', fontsize=14)
    
    # Set X-axis labels
    ax.set_xticks(range(len(resolutions)))
    ax.set_xticklabels(resolutions, rotation=45, ha='right')
    
    # Add values on bars
    for bar, count, percentage in zip(bars, counts, percentages):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + height * 0.01,
            f'{count}\n({percentage:.1f}%)',
            ha='center',
            va='bottom',
            fontweight='bold',
            fontsize=9
        )
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Top resolutions chart saved as: {save_path}")
    plt.show()

if __name__ == "__main__":
    print("🎯 EXDARK DATASET RESOLUTION ANALYSIS")
    print("=" * 60)
    
    # Set path to the ExDark dataset
    exdark_dataset_path = "ExDark"  # Change to the correct path
    
    if not os.path.exists(exdark_dataset_path):
        print(f"❌ ExDark dataset not found in: {exdark_dataset_path}")
        print("💡 Set the correct path in the 'exdark_dataset_path' variable")
        exit(1)
    
    # Run analysis
    stats = analyze_image_resolutions(exdark_dataset_path)
    
    if stats:
        # Create plots
        print("\n🎨 Creating plots...")
        create_resolution_plots(stats)
        create_top_resolutions_chart(stats)
        
        print("\n✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        print("📁 Output files:")
        print("  • exdark_resolution_analysis.png - full analysis")
        print("  • exdark_top_resolutions.png - top resolutions")
    else:
        print("\n❌ ANALYSIS FAILED")
        print("💡 Check the dataset path and folder structure")