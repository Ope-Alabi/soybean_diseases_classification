import os
import shutil
from pathlib import Path
import random

def extract_subset_from_all(input_root_dir, output_root_dir, extract_ratio=0.45, seed=None):
    # Set the seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Initialize counters
    class_counts_original = {}
    class_counts_subset = {}

    # Process each dataset split (train, val, test)
    for split in ['train', 'val', 'test']:
        input_dir = Path(input_root_dir) / split
        output_dir = Path(output_root_dir) / split
        output_dir.mkdir(parents=True, exist_ok=True)

        # Traverse each class subfolder in the input directory
        for class_dir in os.listdir(input_dir):
            class_path = os.path.join(input_dir, class_dir)
            if os.path.isdir(class_path):
                images = os.listdir(class_path)
                random.shuffle(images)
                
                num_images = len(images)
                extract_count = int(num_images * extract_ratio)

                # Create class subdirectories in the output directory
                (output_dir / class_dir).mkdir(parents=True, exist_ok=True)

                # Select images to extract
                extract_images = images[:extract_count]

                # Copy selected images to the new directory
                for image in extract_images:
                    shutil.copy(os.path.join(class_path, image), output_dir / class_dir / image)

                # Store counts
                class_counts_original.setdefault(split, {})[class_dir] = num_images
                class_counts_subset.setdefault(split, {})[class_dir] = len(extract_images)

                print(f"Processed class '{class_dir}' in '{split}': {num_images} original, {len(extract_images)} extracted.")

    # Calculate total classes and images across all splits
    total_classes_original = sum(len(classes) for classes in class_counts_original.values())
    total_classes_subset = sum(len(classes) for classes in class_counts_subset.values())

    total_images_original = sum(sum(class_counts_original[split].values()) for split in class_counts_original)
    total_images_subset = sum(sum(class_counts_subset[split].values()) for split in class_counts_subset)

    print(f"Original directory: {total_classes_original} classes, {total_images_original} images across all splits.")
    print(f"New subset directory: {total_classes_subset} classes, {total_images_subset} images across all splits.")

    return class_counts_original, class_counts_subset

# Example usage
input_root_dir = '/Users/oalabi1/Desktop/PhD/Datasets/Soybean_ML_orig'
output_root_dir = '/Users/oalabi1/Desktop/PhD/Datasets/Soybean_ML_subset'
# extract_subset_from_all(input_root_dir=input_root_dir, output_root_dir=output_root_dir, seed=12345)


import os

# Specify the path to your data folder
# data_folder = '/Users/oalabi1/Desktop/PhD/Datasets/Soybean_ML_subset/Test'  # Replace with the actual path
data_folder = '/Users/oalabi1/Desktop/PhD/Datasets/PlantDataset_split/Test'
# Define the image file extensions you want to consider
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}

# Get a list of all subdirectories (classes) in the data folder
classes = [d for d in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, d))]

# Iterate over each class and count the number of image files
for class_name in classes:
    class_path = os.path.join(data_folder, class_name)
    # List all files in the class directory
    files = os.listdir(class_path)
    # Filter files to include only image files based on their extensions
    images = [
        f for f in files
        if os.path.isfile(os.path.join(class_path, f)) and os.path.splitext(f)[1].lower() in image_extensions
    ]
    # Count the number of images
    num_images = len(images)
    print(f"Class '{class_name}': {num_images} images")