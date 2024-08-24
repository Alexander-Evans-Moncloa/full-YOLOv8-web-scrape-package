from icrawler.builtin import BingImageCrawler
from datetime import date
import os
import random
import shutil

# Step 0: Define maximum number of images to be crawled and keywords
maxnum = 260
keywords = ['car', 'motorbike', 'kitten', 'puppy']  # Define your keywords here
base_dir = 'Test_data'  # Base directory for storing the data

# Function to crawl images for a given keyword
def crawl_images(keyword, max_num, base_dir):
    dump_dir = os.path.join(base_dir, f"{keyword}_dump")
    bing_crawler = BingImageCrawler(
        feeder_threads=1,
        parser_threads=2,
        downloader_threads=4,
        storage={'root_dir': dump_dir}
    )
    bing_crawler.crawl(keyword=keyword, max_num=max_num)
    return dump_dir

# Function to process and split images into train/val sets
def process_images(keyword, source_directory, base_dir):
    train_directory = os.path.join(base_dir, 'train', keyword)
    val_directory = os.path.join(base_dir, 'val', keyword)

    # Ensure destination directories exist
    os.makedirs(train_directory, exist_ok=True)
    os.makedirs(val_directory, exist_ok=True)

    # List all image files (filter to only include .jpg)
    all_images = [f for f in os.listdir(source_directory) if f.lower().endswith('.jpg')]

    # Print diagnostic information
    print(f"Processing {keyword}: {len(all_images)} images found.")

    # Shuffle and split images
    random.shuffle(all_images)
    split_index = int(0.77 * len(all_images))  # 77% for training
    train_images = all_images[:split_index]
    val_images = all_images[split_index:]

    # Move training images
    for image in train_images:
        source_path = os.path.join(source_directory, image)
        destination_path = os.path.join(train_directory, image)
        shutil.move(source_path, destination_path)

    # Move validation images
    for image in val_images:
        source_path = os.path.join(source_directory, image)
        destination_path = os.path.join(val_directory, image)
        shutil.move(source_path, destination_path)

    # Delete the source directory
    shutil.rmtree(source_directory)
    print(f"Completed processing for {keyword}. Source directory deleted.")

# Main loop to crawl and process images for each keyword
for keyword in keywords:
    print(f"Starting crawl for {keyword} images...")
    source_directory = crawl_images(keyword, maxnum, base_dir)
    process_images(keyword, source_directory, base_dir)

print("All datasets have been processed and organized successfully!")
