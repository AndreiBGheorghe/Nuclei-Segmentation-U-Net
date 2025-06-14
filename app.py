import streamlit as st
from PIL import Image
import torch
import numpy as np
from torchvision.transforms import ToTensor
from unet import UNet

st.set_page_config(page_title="Nuclei Segmentation", layout="wide")
st.title("Nuclei Segmentation Using U-Net")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
uploaded_file = st.file_uploader("Upload a PNG Image", type=["png"])

@st.cache_resource
def load_model():
    model = UNet().to(device)
    model.load_state_dict(torch.load("unet.pth", map_location=device))
    model.eval()
    return model

model = load_model()

def predict_mask(img: Image.Image):
    img = img.convert("RGB").resize((256, 256))
    img_tensor = ToTensor()(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = torch.sigmoid(model(img_tensor))[0][0].cpu().numpy()
    bin_mask = (output > output.max() / 10).astype(np.uint8) * 255
    return img, bin_mask

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.subheader("Image and Mask")
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Image", width=256)
    with col2:
        _, mask = predict_mask(img)
        st.image(mask, clamp=True, caption="Mask", width=256)