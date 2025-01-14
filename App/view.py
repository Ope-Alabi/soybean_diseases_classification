import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
from torchinfo import summary

import streamlit as st

import time
import os
import copy
import json
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loading the class names
# class_indices = json.load(open("/Users/oalabi1/Library/CloudStorage/OneDrive-UniversityofNorthCarolinaatCharlotte/Personal Projects/Machine Learning/soybean_diseases_classification/App/class_indices.json"))
class_indices = json.load(open("C:/Users/alabi/OneDrive - University of North Carolina at Charlotte/Personal Projects/Machine Learning/soybean_diseases_classification/App/class_indices.json"))
num_classes = len(class_indices)

# Load class names from JSON file
# with open("/Users/oalabi1/Library/CloudStorage/OneDrive-UniversityofNorthCarolinaatCharlotte/Personal Projects/Machine Learning/soybean_diseases_classification/App/class_indices.json", 'r') as f:
#     class_names = json.load(f)

with open("C:/Users/alabi/OneDrive - University of North Carolina at Charlotte/Personal Projects/Machine Learning/soybean_diseases_classification/App/class_indices.json", 'r') as f:
    class_names = json.load(f)




# Load the model architecture
model = models.densenet201(weights=None).to(device)

# Load the pre-trained weights
# model_path = "C:/Users/alabi/OneDrive - University of North Carolina at Charlotte/Personal Projects/Machine Learning/Plant Disease Prediction/model_training/base_models/ResNet50-Plant-model-80.pth"
# model_path = "/Users/oalabi1/Library/CloudStorage/OneDrive-UniversityofNorthCarolinaatCharlotte/Personal Projects/Machine Learning/soybean_diseases_classification/OUTPUT/OUTPUT/SAVED_MODELS/densenet201_model[pddd=False,frozen=False,pretrained=True].pth"
model_path = "C:/Users/alabi/OneDrive - University of North Carolina at Charlotte/Personal Projects/Machine Learning/soybean_diseases_classification/OUTPUT/OUTPUT/SAVED_MODELS/densenet201_model[pddd=False,frozen=False,pretrained=True].pth"

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








# # Step 1: Load the pretrained DenseNet model
# # model = models.densenet201(pretrained=False)
# model = models.densenet201(weights=None).to(device)

# # Load the model's state_dict
# model_path = "/Users/oalabi1/Library/CloudStorage/OneDrive-UniversityofNorthCarolinaatCharlotte/Personal Projects/Machine Learning/soybean_diseases_classification/OUTPUT/OUTPUT/SAVED_MODELS/densenet201_model[pddd=False,frozen=False,pretrained=True].pth"
# # state_dict = torch.load(model_path, map_location='cpu')
# state_dict = torch.load(model_path, map_location=device)

# # Step 2: Modify the classifier layer
# num_classes = 8
# model.classifier = nn.Sequential(
#     nn.Linear(in_features=model.classifier.in_features, out_features=128, bias=True),
#     nn.ReLU(inplace=True),
#     nn.Dropout(p=0.2),
#     nn.Linear(in_features=128, out_features=64, bias=True),
#     nn.ReLU(inplace=True),
#     nn.Dropout(p=0.3),
#     nn.Linear(in_features=64, out_features=num_classes, bias=True)  # Replace with your number of output classes
# )

# # Step 3: Load the pretrained state_dict, but ignore the classifier weights
# # This method allows you to load the state_dict while ignoring layers that don't match
# model_dict = model.state_dict()
# pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and "classifier" not in k}
# model_dict.update(pretrained_dict)
# model.load_state_dict(model_dict)

# # Optionally freeze model parameters
# for param in model.parameters():
#     param.requires_grad = False


# Move the model to the appropriate device
model.to(device)

# # Print a summary of the model
# summary(model=model,
#         input_size=(32, 3, 224, 224),
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"])

# print(summary)

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

# Load and preprocess the image
def preprocess_img(image_path):
    # image_path = 'path_to_your_image.jpg'
    image = Image.open(image_path)
    image = Image.open(image_path)
    image = data_transforms(image)
    image = image.unsqueeze(0)  # Add batch dimension
    image = image.to(device)
    return image

# def predict_image_class (model, path, class_names):
#     # Make the prediction
#     image = preprocess_img(path)
#     with torch.no_grad():
#         outputs = model(image)
#         print(outputs)
#         print(outputs.shape)

#         # Apply softmax to get the probabilities
#         probabilities = F.softmax(outputs, dim=1)
        
#         # Get the predicted class index and its probability
#         max_prob, preds = torch.max(probabilities, 1) # value and index
#         # print(preds)
#         print(max_prob, probabilities)

#         # Get the predicted class
#         predicted_class = preds.item()
#         predicted_class_name = class_indices[str(predicted_class)]

#         # Convert max_prob to a percentage
#         confidence_score = max_prob.item() * 100

#         return predicted_class_name, confidence_score


# # Streamlit App
# st.title('Plant Disease Classifier')

# uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

# if uploaded_image is not None:
#     image = Image.open(uploaded_image)
#     col1, col2 = st.columns(2)

#     with col1:
#         resized_img = image.resize((150, 150))
#         st.image(resized_img)

#     with col2:
#         if st.button('Classify'):
#             # Preprocess the uploaded image and predict the class
#             prediction, conf_score = predict_image_class(model, uploaded_image, class_indices)
#             st.success(f'Prediction: {str(prediction)} ({str(conf_score)})')



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

st.title('Plant Disease Classifier')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Preprocess the uploaded image and predict the class
            top3_predictions = predict_image_class(model, uploaded_image, class_names)
            
            # Display the top 3 predictions and their confidence scores
            for i, (pred, conf_score) in enumerate(top3_predictions):
                st.success(f'Top {i+1} Prediction: {pred} ({conf_score:.2f}%)')
