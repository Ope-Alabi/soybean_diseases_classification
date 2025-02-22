import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms

import streamlit as st

import time
import os
import copy
import json
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

# implementation of langchain and groq
from langchain_groq import ChatGroq
from langchain.prompts.chat import ChatPromptTemplate

# Set the GROQ API key
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Define the prompt for generating IPM treatment output
DISEASES_RECOMMENDATION_PROMPT = """
Your role is a plant disease expert, the given disease has already be verified, generate IPM [Integrated Pest Management] treatment output using the template below. 

[INFORMATION ABOUT PLANT]
Crop name: {CROP_NAME}
Location: {CROP_LOCATION}
Season: {CROP_SEASON}
Verified disease: {CROP_VERIFIED_DISEASE} 

### OUTPUT TEMPLATE ###
[INSERT A NUMBERED LIST OF TREATMENT OPTIONS]
[CREATE A STEP BY STEP APPLICATION FOR THE 1ST OPTION]
[INSERT A NUMBERED LIST OF IPM TREATMENT OPTIONS]
[CREATE A STEP BY STEP APPLICATION FOR THE 1ST OPTION]
[CITE YOUR TREATMENT INFORMATION]

Examples:

Your role is a plant disease expert; the given disease has already been verified. Generate IPM [Integrated Pest Management] treatment output using the template below.

<INFORMATION ABOUT PLANT>
Crop name: (Soybean)
Location: (North Carolina)
Season: (Dry)
Verified disease: (Cercospora leaf blight)

### OUTPUT TEMPLATE ###
<INSERT A NUMBERED LIST OF TREATMENT OPTIONS>
<CREATE A STEP-BY-STEP APPLICATION FOR THE 1ST OPTION>
<INSERT A NUMBERED LIST OF IPM TREATMENT OPTIONS>
<CREATE A STEP-BY-STEP APPLICATION FOR THE 1ST OPTION>
<CITE YOUR TREATMENT INFORMATION>
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the class indices

class_indices = json.load(open(r"C:\Users\alvaj\Documents\soybean_diseases_classification\App\class_indices.json"))
num_classes = len(class_indices) 

# Load the class names
with open(r"C:\Users\alvaj\Documents\soybean_diseases_classification\App\class_indices.json", 'r') as f:
    class_names = json.load(f)

# Load the model architecture
model = models.densenet201(weights=None).to(device)

# Load the pre-trained weights
# model_path = "C:/Users/alabi/OneDrive - University of North Carolina at Charlotte/Personal Projects/Machine Learning/Plant Disease Prediction/model_training/base_models/ResNet50-Plant-model-80.pth"
# model_path = "/Users/oalabi1/Library/CloudStorage/OneDrive-UniversityofNorthCarolinaatCharlotte/Personal Projects/Machine Learning/soybean_diseases_classification/OUTPUT/OUTPUT/SAVED_MODELS/densenet201_model[pddd=False,frozen=False,pretrained=True].pth"
model_path = r"C:\Users\alvaj\Documents\soybean_diseases_classification\OUTPUT\densenet201_model[pddd=False,frozen=False,pretrained=True].pth"
state_dict = torch.load(model_path, map_location=device)

# Extract weights and biases for each layer in the classifier
fc0_weight = state_dict.pop('classifier.0.weight', None)
fc0_bias = state_dict.pop('classifier.0.bias', None)
fc3_weight = state_dict.pop('classifier.3.weight', None)
fc3_bias = state_dict.pop('classifier.3.bias', None)
fc6_weight = state_dict.pop('classifier.6.weight', None)
fc6_bias = state_dict.pop('classifier.6.bias', None)

# Load the remaining state_dict into the model (excluding the classifier layers)
model.load_state_dict(state_dict, strict=False)

# Replace the model's classifier with a new one
# num_classes = fc6_weight.shape[0] if fc6_weight is not None else 1000  # Replace 1000 with your default number of classes
model.classifier = nn.Sequential(
    nn.Linear(in_features=fc0_weight.shape[1], out_features=fc0_weight.shape[0], bias=True),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.2),
    nn.Linear(in_features=fc3_weight.shape[1], out_features=fc3_weight.shape[0], bias=True),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.3),
    nn.Linear(in_features=fc6_weight.shape[1], out_features=fc6_weight.shape[0], bias=True)
)

# Assign the loaded weights and biases to the new classifier layers
if fc0_weight is not None and fc0_bias is not None:
    model.classifier[0].weight = nn.Parameter(fc0_weight)
    model.classifier[0].bias = nn.Parameter(fc0_bias)

if fc3_weight is not None and fc3_bias is not None:
    model.classifier[3].weight = nn.Parameter(fc3_weight)
    model.classifier[3].bias = nn.Parameter(fc3_bias)

if fc6_weight is not None and fc6_bias is not None:
    model.classifier[6].weight = nn.Parameter(fc6_weight)
    model.classifier[6].bias = nn.Parameter(fc6_bias)

# Optionally freeze model parameters
for param in model.parameters():
    param.requires_grad = False

# Move the model to the appropriate device
model.to(device)

model.eval()

# Define the data transformation
mean = np.array([0.416, 0.468, 0.355]) # From the PDDD paper. Why was this chosen?
std = np.array([0.210, 0.206, 0.213])

data_transforms = transforms.Compose([
    # transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Gerenate IPM treatment output
def generate_treatment_output(crop_name, crop_location, crop_season, crop_verified_disease):
    # Initialize the ChatGroq object
    llm = ChatGroq(api_key=GROQ_API_KEY, model="mixtral-8x7b-32768", temperature=0.5)
    
    # Initialize the ChatPromptTemplate object
    chat_prompt_template = ChatPromptTemplate(
        prompt=DISEASES_RECOMMENDATION_PROMPT,
        messages=[{"role": "system", "content": DISEASES_RECOMMENDATION_PROMPT}]
    )
    
    query = chat_prompt_template.format(CROP_NAME=crop_name,
        CROP_LOCATION=crop_location,
        CROP_SEASON=crop_season,
        CROP_VERIFIED_DISEASE=crop_verified_disease)
    
    # Generate the IPM treatment output
    ipm_treatment_output = llm.invoke(query)
    
    return ipm_treatment_output.content if ipm_treatment_output.content else "No treatment available"


# Load and preprocess the image
def preprocess_img(image_path):
    # image_path = 'path_to_your_image.jpg'
    image = Image.open(image_path)
    image = Image.open(image_path)
    image = data_transforms(image)
    image = image.unsqueeze(0)  # Add batch dimension
    image = image.to(device)
    return image

def predict_image_class(model, path, class_names):
    # Preprocess the image
    image = preprocess_img(path)
    
    # Make the prediction
    with torch.no_grad():
        outputs = model(image)
        
        # Apply softmax to get the probabilities
        probabilities = F.softmax(outputs, dim=1)
        
        # Get the top 3 predicted class indices and their probabilities
        top3_prob, top3_preds = torch.topk(probabilities, 3, dim=1)
        
        # Convert top3_preds to class names and top3_prob to percentages
        top3_predictions = [(class_names[str(idx.item())], prob.item() * 100) for idx, prob in zip(top3_preds[0], top3_prob[0])]
        
        return top3_predictions


def main(): 
    st.title('Soybean Plant Disease Classifier 🌿')

    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    # ask for the location, season, and verified disease
    crop_location = st.text_input("Enter the location of the plant:")
    crop_season = st.selectbox("Select the season:", ["Dry", "Wet", "Spring", "Summer", "Fall", "Winter"])
    
    
    if uploaded_image is not None and crop_location and crop_season:
        image = Image.open(uploaded_image)
        col1, col2 = st.columns(2)

        with col1:
            resized_img = image.resize((150, 150))
            st.image(resized_img)

        with col2:
            top3_predictions = None
            ipm_treatment_output = None
            
            if st.button('Classify'):
                st.session_state.top3_predictions = predict_image_class(model, uploaded_image, class_names)
                for i, (pred, conf_score) in enumerate(st.session_state.top3_predictions):
                    st.success(f'Top {i+1} Prediction: {pred} ({conf_score:.2f}%)')

        # Ensure top3_predictions exists in session state before generating treatment output
        if ("top3_predictions" in st.session_state 
                and st.session_state.top3_predictions 
                and st.session_state.top3_predictions[0][0].lower() != "healthy"):
            if st.button('Generate Treatment Output'):
                crop_name = "Soybean"
                crop_verified_disease = st.session_state.top3_predictions[0][0]
                ipm_treatment_output = generate_treatment_output(crop_name, crop_location, crop_season, crop_verified_disease)
                st.write(ipm_treatment_output)
        
if __name__ == '__main__':
    print(GROQ_API_KEY)
    main()