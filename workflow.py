import os
import requests
from dotenv import load_dotenv
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import requests
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import torch.nn as nn
import os
import numpy as np

load_dotenv()

API_TOKEN = os.getenv("API_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

image_folder = "./top_influenceurs_2024"

processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

legend_labels = {
    "0": "Background",
    "1": "Hat",
    "2": "Hair",
    "3": "Sunglasses",
    "4": "Upper-clothes",
    "5": "Skirt",
    "6": "Pants",
    "7": "Dress",
    "8": "Belt",
    "9": "Left-shoe",
    "10": "Right-shoe",
    "11": "Face",
    "12": "Left-leg",
    "13": "Right-leg",
    "14": "Left-arm",
    "15": "Right-arm",
    "16": "Bag",
    "17": "Scarf"
}

custom_colormap = {
    0: (0, 0, 0),    
    1: (0, 255, 255),   # Jaune - Hat
    2: (0, 165, 255),   # Orange - Hair
    3: (255, 0, 255),   # Magenta - Sunglasses
    4: (0, 0, 255),     # Rouge - Upper-clothes
    5: (255, 255, 0),   # Cyan - Skirt
    6: (0, 255, 0),     # Vert - Pants
    7: (255, 0, 0),     # Bleu - Dress
    8: (128, 0, 128),   # Violet - Belt
    9: (0, 255, 255),   # Jaune - Left-shoe
    10: (255, 140, 0),  # Orange foncé - Right-shoe
    11: (200, 180, 140), # Beige - Face
    12: (200, 180, 140), # Beige - Left-leg
    13: (200, 180, 140), # Beige - Right-leg
    14: (200, 180, 140), # Beige - Left-arm
    15: (200, 180, 140), # Beige - Right-arm
    16: (0, 128, 255),  # Bleu clair - Bag
    17: (255, 20, 147)  # Rose - Scarf
}


# No Rendering WSL
matplotlib.use('Agg') 

def  PredictMask(image):
    # Resizing Image and normalize pixel (0 to 1) and convert to tensor Pytorch
    inputs = processor(images=image, return_tensors="pt")

    # Outputs per pixel for each clothing class
    outputs = model(**inputs)
    logits = outputs.logits.cpu()

    # Resize tensor to fit original image size
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )

    predicted_mask = upsampled_logits.argmax(dim=1)[0].numpy()

    return predicted_mask

def displayLabels(predicted_mask):
    labels = np.unique(predicted_mask)
    present_labels = [int(label) for label in labels if label != 0 and str(label) in legend_labels]
    present_classes = [legend_labels[str(label)] for label in present_labels]

    ax = plt.gca()

    y_start = 0.9
    line_height = 0.04

    for i, label_id in enumerate(present_labels):
        y = y_start - i * line_height

        rgb = custom_colormap.get(label_id, (255, 255, 255))
        color = tuple(c/255 for c in rgb)

        ax.add_patch(plt.Rectangle((0.05, y-0.02), 0.03, 0.03, color=color, transform=ax.transAxes))

        plt.text(0.1, y - 0.01, present_classes[i], color='white', fontsize=30, transform=ax.transAxes, verticalalignment='center')


def createDisplay(predicted_mask, background, filename):
    # Create Figure
    fig = plt.figure(figsize=(40, 30))
    fig.patch.set_facecolor('black')

    # Display the background mask
    plt.imshow(background)

    colors = [tuple(np.array(rgb)/255) for _, rgb in sorted(custom_colormap.items())]
    cmap = mcolors.ListedColormap(colors)

    # Display the labels with associated colors
    displayLabels(predicted_mask)

    # Display predicted mask
    plt.imshow(predicted_mask, cmap=cmap, alpha=0.5)

    plt.title(filename)
    plt.axis('off')

    return fig  

def saveFig(fig, filename, placeholder):
    fig.savefig(
        f"./results/{filename}_{placeholder}.png",
        bbox_inches='tight',
        pad_inches=0,
        facecolor=fig.get_facecolor()
    )
    plt.close(fig)


def handleImageAndMask(image, mask, filename, placeholder): 
    predicted_mask = PredictMask(image)
    
    fig = createDisplay(predicted_mask, mask, filename)

    saveFig(fig, filename, placeholder)

    return fig


def combineImages(original_img, mask_img, predicted_img, filename):
    size = original_img.size
    mask_img = mask_img.resize(size)
    predicted_img = predicted_img.resize(size)


def loadImages():
    base_folder = "./top_influenceurs_2024"
    image_folder = os.path.join(base_folder, "IMG")
    mask_folder = os.path.join(base_folder, "Mask")

    for filename in os.listdir(image_folder):
        # Load image from path
        image_path = os.path.join(image_folder, filename)
        image = Image.open(image_path).convert("RGB")

        # Load Mask from path
        mask_filename = filename.replace("image_", "mask_")
        mask_path = os.path.join(mask_folder, mask_filename)
        mask = Image.open(mask_path).convert("RGB")
    
        fig_mask = handleImageAndMask(image, mask, filename, "overlay_mask")
        fig_image = handleImageAndMask(image, image, filename, "overlay_image")

        combineImages(image, fig_mask, fig_image, filename)

if __name__ == "__main__":
    loadImages()