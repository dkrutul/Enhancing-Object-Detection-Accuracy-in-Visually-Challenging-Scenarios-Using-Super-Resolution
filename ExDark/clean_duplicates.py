import os
import glob
from collections import defaultdict
import re
import shutil

def extract_index_from_filename(filename):
    """
    Extracts an index from a filename (e.g., 2015_07437 from 2015_07437.png).

    Args:
        filename (str): File name.

    Returns:
        str or None: Extracted index or None if not found.
    """
    # ExDark pattern: YYYY_XXXXX
    pattern = r'(\d{4}_\d{5})'
    match = re.search(pattern, filename)
    return match.group(1) if match else None

def analyze_dataset_indices(images_path, annotations_path):
    """
    Analyzes indices in the ExDark dataset and checks image/annotation matching.

    Args:
        images_path (str): Path to the images folder.
        annotations_path (str): Path to the annotations folder.
    """
    print("🔍 EXDARK INDEX ANALYSIS")
    print("=" * 60)

    exdark_classes = [
        'Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat',
        'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table'
    ]

    total_stats = {
        'total_images': 0,
        'total_annotations': 0,
        'total_matched': 0,
        'total_orphaned_images': 0,
        'total_orphaned_annotations': 0
    }

    for class_name in exdark_classes:
        print(f"\n📁 Class: {class_name}")

        class_images_path = os.path.join(images_path, class_name)
        class_annotations_path = os.path.join(annotations_path, class_name)

        if not (os.path.exists(class_images_path) and os.path.exists(class_annotations_path)):
            print("   ⚠️ Folder does not exist")
            continue

        # === IMAGE ANALYSIS ===
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(glob.glob(os.path.join(class_images_path, '**', ext), recursive=True))

        # Group images by subfolder and extract indices
        image_indices_by_folder = defaultdict(set)
        image_files_by_index = {}

        for img_path in image_files:
            filename = os.path.basename(img_path)
            index = extract_index_from_filename(filename)

            if index:
                # Find subfolder relative to class folder
                rel_path = os.path.relpath(img_path, class_images_path)
                folder = os.path.dirname(rel_path) if os.path.dirname(rel_path) != '.' else 'root'

                image_indices_by_folder[folder].add(index)
                image_files_by_index[index] = img_path

        # === ANNOTATION ANALYSIS ===
        annotation_files = glob.glob(os.path.join(class_annotations_path, '**', '*.txt'), recursive=True)

        # Group annotations by subfolder and extract indices
        annotation_indices_by_folder = defaultdict(set)
        annotation_files_by_index = {}

        for ann_path in annotation_files:
            filename = os.path.basename(ann_path)

            # Remove suffixes: .txt, .png.txt, etc.
            clean_filename = filename
            if clean_filename.endswith('.png.txt'):
                clean_filename = clean_filename[:-8]
            elif clean_filename.endswith('.jpg.txt'):
                clean_filename = clean_filename[:-8]
            elif clean_filename.endswith('.jpeg.txt'):
                clean_filename = clean_filename[:-9]
            elif clean_filename.endswith('.txt'):
                clean_filename = clean_filename[:-4]

            index = extract_index_from_filename(clean_filename)

            if index:
                # Find subfolder relative to class folder
                rel_path = os.path.relpath(ann_path, class_annotations_path)
                folder = os.path.dirname(rel_path) if os.path.dirname(rel_path) != '.' else 'root'

                annotation_indices_by_folder[folder].add(index)
                annotation_files_by_index[index] = ann_path

        # === INDEX COMPARISON ===
        print("   📊 SUBFOLDER STATISTICS:")

        all_image_folders = set(image_indices_by_folder.keys())
        all_annotation_folders = set(annotation_indices_by_folder.keys())
        all_folders = all_image_folders | all_annotation_folders

        class_matched = 0
        class_orphaned_images = 0
        class_orphaned_annotations = 0

        for folder in sorted(all_folders):
            img_indices = image_indices_by_folder.get(folder, set())
            ann_indices = annotation_indices_by_folder.get(folder, set())

            matched = img_indices & ann_indices
            only_images = img_indices - ann_indices
            only_annotations = ann_indices - img_indices

            class_matched += len(matched)
            class_orphaned_images += len(only_images)
            class_orphaned_annotations += len(only_annotations)

            print(f"      📂 {folder}:")
            print(f"         🖼️ Images: {len(img_indices)} indices")
            print(f"         📝 Annotations: {len(ann_indices)} indices")
            print(f"         ✅ Matched: {len(matched)}")

            if only_images:
                print(f"         🚫 Images without annotations: {len(only_images)}")
                # Show first 3 examples
                for idx in sorted(only_images)[:3]:
                    rel_path = os.path.relpath(image_files_by_index[idx], class_images_path)
                    print(f"            - {idx} → {rel_path}")
                if len(only_images) > 3:
                    print(f"            ... and {len(only_images) - 3} more")

            if only_annotations:
                print(f"         🚫 Annotations without images: {len(only_annotations)}")
                # Show first 3 examples
                for idx in sorted(only_annotations)[:3]:
                    rel_path = os.path.relpath(annotation_files_by_index[idx], class_annotations_path)
                    print(f"            - {idx} → {rel_path}")
                if len(only_annotations) > 3:
                    print(f"            ... and {len(only_annotations) - 3} more")

        # Class summary
        total_images = sum(len(indices) for indices in image_indices_by_folder.values())
        total_annotations = sum(len(indices) for indices in annotation_indices_by_folder.values())

        print(f"\n   📈 CLASS SUMMARY {class_name}:")
        print(f"      🖼️ Total images: {total_images}")
        print(f"      📝 Total annotations: {total_annotations}")
        print(f"      ✅ Matched: {class_matched}")
        print(f"      🚫 Images without annotations: {class_orphaned_images}")
        print(f"      🚫 Annotations without images: {class_orphaned_annotations}")
        print(f"      📊 Matching efficiency: {class_matched / max(total_images, 1) * 100:.1f}%")

        # Update global statistics
        total_stats['total_images'] += total_images
        total_stats['total_annotations'] += total_annotations
        total_stats['total_matched'] += class_matched
        total_stats['total_orphaned_images'] += class_orphaned_images
        total_stats['total_orphaned_annotations'] += class_orphaned_annotations

    # === GLOBAL SUMMARY ===
    print(f"\n{'=' * 60}")
    print("📈 GLOBAL SUMMARY:")
    print(f"   🖼️ Total images: {total_stats['total_images']}")
    print(f"   📝 Total annotations: {total_stats['total_annotations']}")
    print(f"   ✅ Matched pairs: {total_stats['total_matched']}")
    print(f"   🚫 Images without annotations: {total_stats['total_orphaned_images']}")
    print(f"   🚫 Annotations without images: {total_stats['total_orphaned_annotations']}")
    print(f"   📊 Global matching efficiency: {total_stats['total_matched'] / max(total_stats['total_images'], 1) * 100:.1f}%")

    # Explain count difference
    images_vs_annotations = total_stats['total_annotations'] - total_stats['total_images']
    print("\n💡 DIFFERENCE EXPLANATION:")
    print(f"   📊 Difference (annotations - images): {images_vs_annotations}")
    print("   🔍 This results from:")
    print(f"      • {total_stats['total_orphaned_annotations']} annotations without matching images")
    print(f"      • {total_stats['total_orphaned_images']} images without matching annotations")

    return total_stats

def remove_orphaned_annotations(images_path, annotations_path, backup_dir="orphaned_annotations_backup", dry_run=True):
    """
    Removes annotations that do not have matching images.

    Args:
        images_path (str): Path to the images folder.
        annotations_path (str): Path to the annotations folder.
        backup_dir (str): Backup folder for removed annotations.
        dry_run (bool): If True, only simulates removal.
    """
    print("\n🗑️ ORPHANED ANNOTATIONS CLEANUP")
    print(f"{'🔍 DRY RUN' if dry_run else '⚠️ ACTUAL REMOVAL'}")
    print("=" * 60)

    if not dry_run:
        # Create backup
        os.makedirs(backup_dir, exist_ok=True)
        print(f"💾 Creating backup in: {backup_dir}")

    exdark_classes = [
        'Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat',
        'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table'
    ]

    total_removed = 0

    for class_name in exdark_classes:
        print(f"\n📁 Cleaning class: {class_name}")

        class_images_path = os.path.join(images_path, class_name)
        class_annotations_path = os.path.join(annotations_path, class_name)

        if not (os.path.exists(class_images_path) and os.path.exists(class_annotations_path)):
            print("   ⚠️ Folder does not exist")
            continue

        # === COLLECT IMAGE INDICES ===
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(glob.glob(os.path.join(class_images_path, '**', ext), recursive=True))

        image_indices = set()
        for img_path in image_files:
            filename = os.path.basename(img_path)
            index = extract_index_from_filename(filename)
            if index:
                image_indices.add(index)

        print(f"   📊 Found {len(image_indices)} unique image indices")

        # === COLLECT ANNOTATIONS AND FIND ORPHANS ===
        annotation_files = glob.glob(os.path.join(class_annotations_path, '**', '*.txt'), recursive=True)

        orphaned_annotations = []

        for ann_path in annotation_files:
            filename = os.path.basename(ann_path)

            # Clean filename
            clean_filename = filename
            if clean_filename.endswith('.png.txt'):
                clean_filename = clean_filename[:-8]
            elif clean_filename.endswith('.jpg.txt'):
                clean_filename = clean_filename[:-8]
            elif clean_filename.endswith('.jpeg.txt'):
                clean_filename = clean_filename[:-9]
            elif clean_filename.endswith('.txt'):
                clean_filename = clean_filename[:-4]

            index = extract_index_from_filename(clean_filename)

            if index and index not in image_indices:
                orphaned_annotations.append((ann_path, index))

        print(f"   🚫 Found {len(orphaned_annotations)} orphaned annotations")

        # === REMOVE ORPHANED ANNOTATIONS ===
        class_removed = 0

        for ann_path, index in orphaned_annotations:
            rel_path = os.path.relpath(ann_path, class_annotations_path)
            print(f"      ❌ {index} → {rel_path}")

            if not dry_run:
                # Backup before removal
                backup_path = os.path.join(backup_dir, class_name, rel_path)
                os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                shutil.copy2(ann_path, backup_path)

                # Remove file
                os.remove(ann_path)

            class_removed += 1

        total_removed += class_removed
        print(f"   ✅ {'Found' if dry_run else 'Removed'} {class_removed} annotations")

    print("\n✅ CLEANUP COMPLETED")
    print(f"📊 {'Found' if dry_run else 'Removed'} total of {total_removed} orphaned annotations")

    if dry_run:
        print("\n💡 To actually remove annotations, run:")
        print("   remove_orphaned_annotations(images_path, annotations_path, dry_run=False)")
    else:
        print(f"💾 Backup created in: {backup_dir}")

    return total_removed

def find_problematic_files(images_path, annotations_path, class_name, max_examples=5):
    """
    Detailed analysis of problematic files for a selected class.

    Args:
        images_path (str): Path to images.
        annotations_path (str): Path to annotations.
        class_name (str): Class name to analyze.
        max_examples (int): Maximum number of examples to display.
    """
    print(f"\n🔍 DETAILED CLASS ANALYSIS: {class_name}")
    print("=" * 50)

    class_images_path = os.path.join(images_path, class_name)
    class_annotations_path = os.path.join(annotations_path, class_name)

    # Collect all indices
    image_indices = set()
    annotation_indices = set()

    # Images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(glob.glob(os.path.join(class_images_path, '**', ext), recursive=True))

    for img_path in image_files:
        filename = os.path.basename(img_path)
        index = extract_index_from_filename(filename)
        if index:
            image_indices.add(index)

    # Annotations
    annotation_files = glob.glob(os.path.join(class_annotations_path, '**', '*.txt'), recursive=True)

    for ann_path in annotation_files:
        filename = os.path.basename(ann_path)

        # Clean filename
        clean_filename = filename
        if clean_filename.endswith('.png.txt'):
            clean_filename = clean_filename[:-8]
        elif clean_filename.endswith('.jpg.txt'):
            clean_filename = clean_filename[:-8]
        elif clean_filename.endswith('.txt'):
            clean_filename = clean_filename[:-4]

        index = extract_index_from_filename(clean_filename)
        if index:
            annotation_indices.add(index)

    # Find issues
    only_images = image_indices - annotation_indices
    only_annotations = annotation_indices - image_indices

    print(f"🚫 IMAGES WITHOUT ANNOTATIONS ({len(only_images)}):")
    for i, index in enumerate(sorted(only_images)[:max_examples]):
        # Find image file
        for img_path in image_files:
            if index in os.path.basename(img_path):
                rel_path = os.path.relpath(img_path, class_images_path)
                print(f"   {i + 1}. {index} → {rel_path}")
                break
    if len(only_images) > max_examples:
        print(f"   ... and {len(only_images) - max_examples} more")

    print(f"\n🚫 ANNOTATIONS WITHOUT IMAGES ({len(only_annotations)}):")
    for i, index in enumerate(sorted(only_annotations)[:max_examples]):
        # Find annotation file
        for ann_path in annotation_files:
            if index in os.path.basename(ann_path):
                rel_path = os.path.relpath(ann_path, class_annotations_path)
                print(f"   {i + 1}. {index} → {rel_path}")
                break
    if len(only_annotations) > max_examples:
        print(f"   ... and {len(only_annotations) - max_examples} more")

if __name__ == "__main__":
    print("🔍 EXDARK DATASET ANALYSIS AND CLEANUP")
    print("=" * 60)

    images_path = "ExDark"
    annotations_path = "ExDark_Annno"

    # Check whether folders exist
    if not os.path.exists(images_path):
        print(f"❌ Images folder not found: {images_path}")
        exit(1)

    if not os.path.exists(annotations_path):
        print(f"❌ Annotations folder not found: {annotations_path}")
        exit(1)

    # 1. Run index analysis
    stats = analyze_dataset_indices(images_path, annotations_path)

    # 2. If orphaned annotations exist, propose removal
    if stats['total_orphaned_annotations'] > 0:
        print(f"\n{'=' * 60}")
        print("🗑️ ORPHANED ANNOTATIONS CLEANUP")

        # Dry-run removal
        removed = remove_orphaned_annotations(images_path, annotations_path, dry_run=True)

        if removed > 0:
            # Ask user for confirmation
            print(f"\n{'=' * 60}")
            answer = input(f"🤔 Remove {removed} orphaned annotations? (y/N): ").lower().strip()

            if answer in ['y', 'yes']:
                # Actual removal
                remove_orphaned_annotations(images_path, annotations_path, dry_run=False)

                # Re-run analysis after cleanup
                print(f"\n{'=' * 60}")
                print("📊 RE-ANALYSIS AFTER CLEANUP")
                analyze_dataset_indices(images_path, annotations_path)
            else:
                print("❌ Removal canceled")

    # 3. Detailed analysis for problematic classes
    if stats['total_orphaned_images'] > 0:
        print(f"\n{'=' * 60}")
        print("🔍 DETAILED ANALYSIS OF IMAGES WITHOUT ANNOTATIONS")

        # Select a few classes for detailed analysis
        problem_classes = ['Bicycle', 'Car', 'Chair']

        for class_name in problem_classes:
            find_problematic_files(images_path, annotations_path, class_name)

    print("\n🎉 ANALYSIS COMPLETED!")