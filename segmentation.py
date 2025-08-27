# utils/segmentation.py
import torch
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2

# Global variables to store loaded models
_deeplab_model = None
_maskrcnn_model = None


def load_deeplab_model():
    """Load DeepLabV3 model once and reuse it"""
    global _deeplab_model
    if _deeplab_model is None:
        print("Loading DeepLabV3 model...")
        _deeplab_model = torchvision.models.segmentation.deeplabv3_resnet50(
            pretrained=True
        )
        _deeplab_model.eval()
        print("DeepLabV3 model loaded successfully!")


def load_maskrcnn_model():
    """Load Mask R-CNN model once and reuse it"""
    global _maskrcnn_model
    if _maskrcnn_model is None:
        print("Loading Mask R-CNN model...")
        _maskrcnn_model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained=True
        )
        _maskrcnn_model.eval()
        print("Mask R-CNN model loaded successfully!")


def perform_segmentation(image_array, model_type="DeepLabV3", threshold=0.5):
    """
    Perform segmentation on the given image

    Args:
        image_array: numpy array of the image
        model_type: either "DeepLabV3" or "Mask R-CNN"
        threshold: confidence threshold for detection

    Returns:
        dict: segmentation results including masks, labels, and scores
    """
    try:
        if model_type == "DeepLabV3":
            return _deeplabv3_segmentation(image_array)
        else:
            return _mask_rcnn_segmentation(image_array, threshold)
    except Exception as e:
        return {"error": str(e)}


def _deeplabv3_segmentation(image_array):
    """Perform segmentation using DeepLabV3"""
    # Load the model if not already loaded
    load_deeplab_model()

    # Preprocess the image
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.fromarray(image_array)
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze((0))

    # Perform prediction
    with torch.no_grad():
        output = _deeplab_model(input_batch)["out"][0]

    # Process the output
    output_predictions = output.argmax(0).byte().cpu().numpy()

    # Get the number of unique classes (excluding background)
    unique_classes = np.unique(output_predictions)
    unique_classes = unique_classes[unique_classes != 0]  # Remove background

    # COCO class names for semantic segmentation
    coco_names = [
        "background",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "street sign",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "hat",
        "backpack",
        "umbrella",
        "shoe",
        "eye glasses",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "plate",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "mirror",
        "dining table",
        "window",
        "desk",
        "toilet",
        "door",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "blender",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
        "hair brush",
    ]

    # Get detected classes
    detected_classes = [coco_names[i] for i in unique_classes if i < len(coco_names)]

    return {
        "masks": output_predictions,
        "labels": detected_classes,
        "model_type": "DeepLabV3",
        "unique_classes": unique_classes.tolist(),
    }


def _mask_rcnn_segmentation(image_array, threshold):
    """Perform instance segmentation using Mask R-CNN"""
    # Load the model if not already loaded
    load_maskrcnn_model()

    # Preprocess the image
    image = Image.fromarray(image_array)
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(image)

    # Perform prediction
    with torch.no_grad():
        prediction = _maskrcnn_model([img_tensor])

    # Filter predictions based on threshold
    masks = prediction[0]["masks"] > threshold
    scores = prediction[0]["scores"].cpu().numpy()
    labels = prediction[0]["labels"].cpu().numpy()

    # Get COCO class names
    coco_names = [
        "background",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "N/A",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "N/A",
        "backpack",
        "umbrella",
        "N/A",
        "N/A",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "N/A",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "N/A",
        "dining table",
        "N/A",
        "N/A",
        "toilet",
        "N/A",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "N/A",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

    # Filter by threshold
    keep = scores >= threshold
    masks = masks[keep]
    scores = scores[keep]
    labels = labels[keep]

    label_names = [coco_names[i] for i in labels]

    return {
        "masks": masks.cpu().numpy(),
        "labels": label_names,
        "scores": scores,
        "model_type": "Mask R-CNN",
    }
