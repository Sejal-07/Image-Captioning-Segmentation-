# utils/visualization.py
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches


def visualize_segmentation(image_array, segmentation_result):
    """
    Create a visualization of the segmentation results

    Args:
        image_array: original image as numpy array
        segmentation_result: result dictionary from segmentation functions

    Returns:
        numpy array: visualization image
    """
    if "error" in segmentation_result:
        return image_array

    if segmentation_result["model_type"] == "DeepLabV3":
        return _visualize_semantic_segmentation(image_array, segmentation_result)
    else:
        return _visualize_instance_segmentation(image_array, segmentation_result)


def _visualize_semantic_segmentation(image_array, segmentation_result, alpha=0.5):
    """
    Overlay semantic segmentation mask on the original image.
    """
    # Get mask from result
    masks = segmentation_result.get("masks")

    if masks is None:
        return image_array

    # If mask has probabilities, take argmax
    if masks.ndim == 3 and masks.shape[2] > 1:
        masks = np.argmax(masks, axis=2)

    # Resize mask to match input image size
    masks = cv2.resize(
        masks.astype(np.uint8),
        (image_array.shape[1], image_array.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    # Generate a colored mask
    unique_classes = np.unique(masks)
    colored_mask = np.zeros_like(image_array, dtype=np.uint8)
    legend_patches = []

    for class_id in unique_classes:
        if class_id == 0:  # background
            continue

        # Pick a distinct color from matplotlib colormap
        color = cm.tab20(class_id % 20)[:3]
        color = tuple(int(c * 255) for c in color)

        colored_mask[masks == class_id] = color

        if "labels" in segmentation_result and class_id - 1 < len(
            segmentation_result["labels"]
        ):
            label = segmentation_result["labels"][class_id - 1]
            legend_patches.append(
                mpatches.Patch(color=np.array(color) / 255, label=label)
            )

    # Blend with original image
    visualization = cv2.addWeighted(image_array, 1 - alpha, colored_mask, alpha, 0)

    segmentation_result["legend_patches"] = legend_patches
    return visualization


def _visualize_instance_segmentation(image_array, segmentation_result):
    """Visualize instance segmentation results"""
    visualization = image_array.copy()
    masks = segmentation_result["masks"]
    labels = segmentation_result["labels"]
    scores = segmentation_result["scores"]

    legend_patches = []

    for i, (mask, label, score) in enumerate(zip(masks, labels, scores)):
        color = cm.tab10(i % 10)[:3]
        color = tuple(int(c * 255) for c in color)

        mask = mask[0] if mask.ndim == 3 else mask

        if mask.shape != image_array.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (image_array.shape[1], image_array.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(visualization, contours, -1, color, 2)

        legend_text = f"{label} ({score:.2f})"
        legend_patches.append(
            mpatches.Patch(color=np.array(color) / 255, label=legend_text)
        )

        if len(contours) > 0:
            x, y, w, h = cv2.boundingRect(contours[0])
            cv2.putText(
                visualization,
                legend_text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    segmentation_result["legend_patches"] = legend_patches
    return visualization


def create_legend_figure(legend_patches):
    """Create a matplotlib figure for the legend"""
    if not legend_patches:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.legend(handles=legend_patches, loc="center", ncol=2)
    ax.axis("off")
    return fig
