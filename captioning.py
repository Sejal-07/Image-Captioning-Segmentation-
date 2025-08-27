# utils/captioning.py
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Global variables to store loaded models (for performance)
_blip_processor = None
_blip_model = None


def load_blip_model():
    """Load BLIP model once and reuse it"""
    global _blip_processor, _blip_model
    if _blip_processor is None or _blip_model is None:
        print("Loading BLIP model...")
        _blip_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        _blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        print("BLIP model loaded successfully!")


def generate_caption(image_array, model_type="BLIP"):
    """
    Generate a caption for the given image using the specified model

    Args:
        image_array: numpy array of the image
        model_type: either "BLIP" or "CLIP"

    Returns:
        str: generated caption
    """
    # Convert numpy array to PIL Image
    image = Image.fromarray(image_array)

    if model_type == "BLIP":
        return _generate_caption_blip(image)
    else:
        return _generate_caption_clip(image)


def _generate_caption_blip(image):
    """Generate caption using BLIP model"""
    try:
        # Load model if not already loaded
        load_blip_model()

        # Process image and generate caption
        inputs = _blip_processor(image, return_tensors="pt")

        # Generate caption with different strategies
        with torch.no_grad():
            # Method 1: Beam search (higher quality)
            out = _blip_model.generate(
                **inputs, max_length=50, num_beams=5, early_stopping=True
            )

            # Method 2: Sampling (more diverse)
            # out = _blip_model.generate(
            #     **inputs,
            #     max_length=50,
            #     do_sample=True,
            #     top_p=0.9,
            #     temperature=0.7
            # )

        caption = _blip_processor.decode(out[0], skip_special_tokens=True)

        return caption.capitalize()

    except Exception as e:
        return f"Error generating caption: {str(e)}"


def _generate_caption_clip(image):
    """Generate caption using CLIP model (simplified version)"""
    try:
        # For a real implementation, we would use CLIP with a GPT model
        # This is a simplified placeholder that could be expanded
        return (
            "A detailed caption would be generated with a full CLIP+GPT implementation."
        )

    except Exception as e:
        return f"Error generating caption: {str(e)}"


def generate_multiple_captions(image_array, num_captions=3):
    """
    Generate multiple diverse captions for the same image

    Args:
        image_array: numpy array of the image
        num_captions: number of captions to generate

    Returns:
        list: list of generated captions
    """
    image = Image.fromarray(image_array)
    load_blip_model()

    captions = []
    try:
        inputs = _blip_processor(image, return_tensors="pt")

        for _ in range(num_captions):
            with torch.no_grad():
                out = _blip_model.generate(
                    **inputs,
                    max_length=50,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.8 + 0.1 * _,  # Vary temperature slightly
                )

            caption = _blip_processor.decode(out[0], skip_special_tokens=True)
            captions.append(caption.capitalize())

    except Exception as e:
        captions = [f"Error generating caption: {str(e)}"]

    return captions
