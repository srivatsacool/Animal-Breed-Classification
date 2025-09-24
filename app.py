import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import torch
import pickle
import base64
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import streamlit_shadcn_ui as ui


# Page setup

st.set_page_config(page_title="BreedDex🐄", layout="wide")

global class_names, device , help_text , class_name, confidence , last_model 
if ["class_name", "confidence" , "last_model","input_img"] not in st.session_state:
    st.session_state.class_name = "__"
    st.session_state.confidence = "__"
    st.session_state.last_model = "__"
    st.session_state.input_img = None
    st.session_state.output_img = None

# Sidebar
st.sidebar.title("Cattle Breed Classification")
st.sidebar.write("🐄 Cattle vs Buffalo")
st.sidebar.markdown("---")
st.sidebar.write("Models:")
st.sidebar.checkbox("SqueezeNet (Classification)", True)
st.sidebar.checkbox("YOLOv11 (Pose Estimation)", True)
st.sidebar.checkbox("ArUco (Size Estimation)", True)
st.sidebar.markdown("---")

# page_bg_gradient = """
# <style>
# [data-testid="stAppViewContainer"] {
#     background: linear-gradient(135deg, #89f7fe 0%, #66a6ff 100%);
# }
# </style>
# """
# st.markdown(page_bg_gradient, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>BREED DEX 🐄🐃</h1>", unsafe_allow_html=True)



def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="1000" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def draw_img(img , text):
    # Overwrite the image (add title/annotation)
    draw = ImageDraw.Draw(img)
    title_text = "Overwritten!"
    try:
        font = ImageFont.truetype("arial.ttf", 32)
    except:
        font = ImageFont.load_default()
    draw.text((10, 10), title_text, fill="red", font=font)

    st.image(img, caption="Overwritten Image", use_column_width=True)
    return img


def preprocess_image(img_path, device):
    # 1. Open & convert to RGB
    img = Image.open(img_path).convert("RGB")

    # 2. Resize (shortest side = 256, keep aspect ratio)
    w, h = img.size
    if w < h:
        new_w, new_h = 256, int(256 * h / w)
    else:
        new_w, new_h = int(256 * w / h), 256
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # 3. Center crop to 224x224
    left = (img.width - 224) // 2
    top = (img.height - 224) // 2
    right = left + 224
    bottom = top + 224
    img = img.crop((left, top, right, bottom))

    # 4. Convert to NumPy array & normalize to [0,1]
    img = np.array(img).astype(np.float32) / 255.0

    # 5. Normalize with ImageNet mean & std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    # 6. Convert HWC -> CHW & to tensor
    img = np.transpose(img, (2, 0, 1))  # (C,H,W)
    tensor = torch.tensor(img).unsqueeze(0).to(device)  # (1,C,H,W)

    return tensor

# Load the model using pickle
def use_squzze(img):
    print("now running squeeze net")
    info_box.info("Squeeze Net Model is loading...")
    with open('SqueezeNet_model_pickle1.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    # Use loaded_model for prediction
    loaded_model.eval()
    loaded_model.to(device)
    # Change to your image path
    # img = Image.open(img).convert('RGB')
    # transform = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    # ])
    # input_tensor_img= transform(img).unsqueeze(0).to(device)
    input_tensor_img= preprocess_image(img, device)

    with torch.no_grad():
        output = loaded_model(input_tensor_img)
        pred_list = output.sort(dim=1, descending=True).indices[0].tolist()
        probs = torch.nn.functional.softmax(output, dim=1)
        # Convert probs to list of confidence values rounded to 3 decimals
        confidence_list = [round(x, 5) for x in probs[0].tolist()]
        print('Confidence list:', confidence_list)
        
    return (pred_list, confidence_list)


def run_main_pipline(img_file):
    print("running  main pipline")

    pred_list, confidence_list  = use_squzze(img_file)


    st.session_state.class_name = class_names[pred_list[0]]
    plt.title(f"Predicted class index: {class_names[pred_list[0]]}")
    st.session_state.confidence = (confidence_list[pred_list[0]])*100
    st.session_state.last_model = "SqueezeNet"

    #class_res = st.write(f"Classification Result: =  **{st.session_state.class_name}** ")


    #st.session_state.confidence = confidence[pred_class].item()

    print("done with Squeeze net")
    print(f"Image classified as {st.session_state.class_name}!")
    st.success(f"Image classified as **{st.session_state.class_name}** , with Confidence: = **{st.session_state.confidence} %**")
    st.info("Step 1 : Squeeze Net Model is Done .\n\n Step 2 : Yolo Pose Estimation now loading...")
    info_box.info("Pipeline finished!")
    # show_pdf(f".\Breeds\\{st.session_state.class_name}.pdf")


# Breed class names
class_names = ['Alambadi', 'Amritmahal', 'Ayrshire', 'Banni', 'Bargur', 'Bhadawari', 'Brown_Swiss', 'Dangi', 'Deoni', 'Gir', 'Guernsey', 'Hallikar', 'Hariana', 'Holstein_Friesian', 'Jaffrabadi', 'Jersey', 'Kangayam', 'Kankrej', 'Kasargod', 'Kenkatha', 'Kherigarh', 'Khillari', 'Krishna_Valley', 'Malnad_gidda', 'Mehsana', 'Murrah', 'Nagori', 'Nagpuri', 'Nili_Ravi', 'Nimari', 'Ongole', 'Pulikulam', 'Rathi', 'Red_Dane', 'Red_Sindhi', 'Sahiwal', 'Surti', 'Tharparkar', 'Toda', 'Umblachery', 'Vechur']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Title
# st.title("🐄🐃 Animal Type Classification")

# Upload

# Layout with 2 columns
col1, col2 = st.columns([2, 1],gap="large")
# Create a placeholder for info messages
info_box = st.empty()

with col1:
    #st.subheader("Input Image")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.session_state.input_img = uploaded_file
        st.session_state.output_img = uploaded_file
        photo = st.image(st.session_state.output_img , use_container_width=True)
        if st.button("Run Pipeline"):
            st.toast("Running pipeline...",icon=":material/thumb_up:",duration="short")
            info_box.info("Running pipeline...")
            run_main_pipline(uploaded_file)
            # info_box.info("Pipeline finished!")
            st.toast("Pipeline Completed...",icon=":material/thumb_up:")
            show_pdf(f".\Breeds\\{st.session_state.class_name}.pdf")
            
    else:
        info_box.info("👆 Upload an image to get started.")
    

with col2:
    

    
    class_name, confidence , last_model = "--", "--", "--"

    st.subheader("Results")
    class_res = st.write(f"**Classification Result:** = **{st.session_state.class_name}** ")
    st.write(f"**Confidence:** = **{st.session_state.confidence}** %")
    last_model = st.write(f"**Last Model Used:** = **{st.session_state.last_model}**")
    st.write("**Estimated Size:** =**  cm**")

# Footer
st.markdown("---")
st.text("Prototype UI • Srivatsa Gorti")

