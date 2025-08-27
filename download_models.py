# models/download_models.py
import torch
import torchvision
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
import requests
from tqdm import tqdm


def download_coco_dataset_sample():
    """Download a small sample of COCO dataset for testing"""
    print("Downloading sample COCO dataset...")

    # Create directories
    os.makedirs("data/coco/images", exist_ok=True)
    os.makedirs("data/coco/annotations", exist_ok=True)

    # Sample image URLs from COCO dataset
    sample_images = [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        "http://images.cocodataset.org/val2017/000000039770.jpg",
        "http://images.cocodataset.org/val2017/000000039771.jpg",
    ]

    # Download sample images
    for i, url in enumerate(sample_images):
        try:
            response = requests.get(url, stream=True)
            with open(f"data/coco/images/sample_{i+1}.jpg", "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded sample image {i+1}")
        except Exception as e:
            print(f"Error downloading sample image: {e}")

    print("Sample dataset download complete!")


def download_models():
    """Download all required models with progress tracking"""
    print("Downloading models...")

    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    try:
        # Download segmentation models
        print("Downloading DeepLabV3...")
        torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)

        print("Downloading Mask R-CNN...")
        torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

        # Download captioning models
        print("Downloading BLIP processor...")
        BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

        print("Downloading BLIP model...")
        BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )

        print("All models downloaded successfully!")

        # Download sample dataset
        download_coco_dataset_sample()

    except Exception as e:
        print(f"Error downloading models: {e}")
        print("Please check your internet connection and try again.")


if __name__ == "__main__":
    download_models()
