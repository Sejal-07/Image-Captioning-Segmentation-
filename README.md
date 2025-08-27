# 🖼️ Image Captioning & Segmentation App

An interactive **Streamlit-based web app** that performs **image captioning** and **semantic segmentation** using deep learning models.  
Upload an image, and the app will generate a descriptive caption along with a segmentation map that highlights detected objects/classes.

---

## 🚀 Features
- ✅ **Automatic Image Captioning** – Generates natural language descriptions for uploaded images.  
- ✅ **Semantic Segmentation** – Detects objects and overlays colored masks for better visualization.  
- ✅ **Interactive Web UI** – Built with [Streamlit](https://streamlit.io/).  
- ✅ **Object/Class Detection** – Displays detected objects in the image.  
- ✅ **Real-time Processing** – Upload → Analyze → View Results instantly.

---

## 📂 Project Structure
```bash
IMAGE-CAPTIONING-SEGMENTATION/
├── data/
│   └── coco/
│       └── images/
│           ├── sample_image1.png
│           ├── sample_image2.jpeg
│           └── sample_image3.jpeg
│
├── models/                     # Pretrained / trained models will be stored here
│
├── utils/
│   ├── __init__.py             # Marks utils as a package
│   ├── captioning.py           # Image captioning module
│   ├── segmentation.py         # Image segmentation module
│   └── visualization.py        # Visualization utilities
│
├── app.py                      # Main application entry point
├── requirements.txt            # Python dependencies
├── setup.py                    # Project setup script

```

---

## ⚙️ Installation

1. **Clone the repository**  
```bash
git clone https://github.com/Sejal-07/mage-captioning-segmentation.git
cd image-captioning-segmentation
```
2. **Install dependencies**
```bash
pip install -r requirements.txt
```
4. **Usage**
```
streamlit run app.py
```
---

## Example Image
Input Image: 

![sample_image2](https://github.com/user-attachments/assets/999a3b83-6220-431c-8e47-3c0d3f35536a)

---
Result:

<img width="1310" height="535" alt="Screenshot 2025-08-27 191731" src="https://github.com/user-attachments/assets/68b64557-701e-4f6b-b205-01d9e841865b" />
---

## 🏁 Credits
Developed by Sejal Dabre
Project • 2024-25
