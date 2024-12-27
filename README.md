#Meesho: Visual Taxonomy Challenge

This repository contains a Python implementation for predicting product attributes from images, specifically tailored for the "Men T-Shirts" category. The project uses a combination of pre-trained Vision Transformers (ViT) and Neural Networks for feature extraction, imputation, and classification.

##Table of Contents

Overview

Technologies Used

Setup Instructions

Data Processing

Feature Extraction

Model Training

Prediction and Evaluation

###Overview

The goal of this project is to predict product attributes (e.g., color, sleeve type) for "Men T-Shirts" using:

Image data (product photos)

Tabular data (product attributes)

This involves:

Image feature extraction using Vision Transformers (ViT).

One-hot encoding and imputation for missing attributes.

Classification using a custom neural network.

###Technologies Used

Python

PyTorch

TensorFlow/Keras

OpenCV

HuggingFace Transformers

Scikit-learn

Pandas, NumPy

###Setup Instructions

Clone the Repository:

git clone <repository-url>
cd <repository-folder>

Install Dependencies:
Install required libraries using pip:

pip install -r requirements.txt

###Attribute Encoding:

Product attributes are one-hot encoded for easy integration into the machine learning pipeline.

###KNN Imputation:

Missing attributes are imputed using the KNNImputer to fill gaps in the data.

Feature Extraction

###Vision Transformer (ViT):

A pre-trained ViT model (google/vit-base-patch16-224-in21k) is used to extract features from images.

The CLS token representation from the last hidden state is used as the feature vector for each image.

###Normalization:

Extracted features are normalized using the Normalizer from Scikit-learn for model compatibility.

###Model Training

####Custom Neural Network:

A dense feedforward neural network is built using TensorFlow/Keras with the following layers:

Dense layers (512, 256, 128 units)

Dropout for regularization

Batch normalization for stability

Output layer with sigmoid activation for multi-label classification.

###Training Configuration:

Optimizer: Adam

Loss: Binary Crossentropy

Metrics: Accuracy

###Key Functions

reverse_attr_1 to reverse_attr_5: Map one-hot encoded attributes back to their categorical values.

KNN Imputation: Handle missing attribute values efficiently.

###Outputs

Final predictions are saved in a DataFrame containing:

id: Image IDs

Category: Product categories

Predicted attributes (attr_1 to attr_5)

###Acknowledgments

Dataset: Kaggle Visual Taxonomy Challenge

Pre-trained Models: HuggingFace Transformers

Frameworks: PyTorch, TensorFlow, Scikit-learn
