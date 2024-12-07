# Full YOLOv8 Web Scrape Package

A complete pipeline to go from zero data to a trained and tested computer vision machine learning model using YOLOv8. This package automates image scraping, training, and testing while visualizing results, providing an efficient and streamlined process for building a custom image classification model.

---

## Features

1. **Automated Image Collection**:  
   Scrapes approximately 200 images per category for any four user-defined categories using Bing and `icrawler`.

2. **Custom YOLOv8 Model Training**:  
   Utilizes the scraped images to train a YOLOv8 model tailored for your specific categories.

3. **Visualization of Training Metrics**:  
   Leverages `matplotlib` to display metrics such as loss and accuracy across epochs.

4. **Model Testing**:  
   Tests the trained model on four provided images, displaying predictions and their confidence scores.

---

## Installation

Ensure you have Python installed, then install the required dependencies:

```bash
pip install ultralytics
pip install matplotlib
pip install pandas
pip install cv2
pip install icrawler
```

---

## Scripts Overview

1. **`image_downloader.py`**  
   - Uses `icrawler` to scrape images from Bing for four specified categories.
   - Saves the images in a structured directory for training and testing.  
   - Input: Four category names.

2. **`YOLOimageclassifier.py`**  
   - Creates a YOLOv8 model using the images downloaded.  
   - Handles dataset preparation, training, and saving the trained model.  

3. **`results.py`**  
   - Visualizes model performance metrics such as loss and accuracy over training epochs using `matplotlib`.  

4. **`visualise_result.py`**  
   - Tests the trained YOLOv8 model on four sample images.  
   - Displays predictions with confidence percentages.

---

## Usage

1. **Step 1: Scrape Images**  
   Run `image_downloader.py` to scrape and save images for your categories:  
   ```bash
   python image_downloader.py
   ```
   Specify the four categories when prompted.

2. **Step 2: Train the Model**  
   Run `YOLOimageclassifier.py` to train a YOLOv8 model on the downloaded images:  
   ```bash
   python YOLOimageclassifier.py
   ```

3. **Step 3: Visualize Training Results**  
   Run `results.py` to see the model's training metrics:  
   ```bash
   python results.py
   ```

4. **Step 4: Test the Model**  
   Run `visualise_result.py` to test the model on four sample images:  
   ```bash
   python visualise_result.py
   ```

---

## Example Workflow

1. **Image Collection**: Specify four categories, e.g., `cats`, `dogs`, `cars`, `bikes`.  
2. **Model Training**: The YOLOv8 model is trained on the scraped images, with results saved to a file.  
3. **Result Visualization**: View plots of loss and accuracy to assess training progress.  
4. **Model Testing**: Test on sample images to see predictions and confidence scores.

---

## Requirements

- Python 3.8 or higher  
- Internet connection (for image scraping)

---

## Repository Structure

```
full-YOLOv8-web-scrape-package/
├── image_downloader.py      # Scrapes and saves images
├── YOLOimageclassifier.py   # Trains YOLOv8 model
├── results.py               # Visualizes training metrics
├── visualise_result.py      # Tests and visualizes predictions
├── README.md                # Project documentation
```

---

## Notes

- Images scraped are public domain but ensure compliance with copyright laws.  
- Modify the scripts to customize image categories or dataset size.  

---

## Acknowledgments

This project leverages the following libraries and frameworks:  
- **[Ultralytics](https://github.com/ultralytics)** for YOLOv8.  
- **[Matplotlib](https://matplotlib.org/)** for data visualization.  
- **[iCrawler](https://github.com/hellock/icrawler)** for web scraping.  

Happy Training! 🎉 
