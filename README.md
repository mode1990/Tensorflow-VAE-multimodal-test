
# Multi-Omics Data Integration with Variational Autoencoders (VAE) and SHAP Analysis

This project demonstrates how to integrate multi-omics data using a Variational Autoencoder (VAE) with a classification component. The process includes dimensionality reduction and feature importance analysis using Principal Component Analysis (PCA) and SHAP (SHapley Additive exPlanations).

## Steps

1. **Data Generation and Preprocessing:**
   - **Synthetic Data Generation**: Creates synthetic multi-omics datasets (scRNA-seq, scATAC-seq, cell painting).
   - **Data Combination**: Merges the different data types into a single dataset.
   - **Data Normalization**: Normalizes the combined dataset to ensure consistency.

2. **VAE Model:**
   - **Architecture**: A VAE model with an encoder, decoder, and classifier is defined. The encoder learns latent representations, the decoder reconstructs the input data, and the classifier performs binary classification.
   - **Custom Loss Function**: The model is trained using a custom loss function that combines reconstruction loss, KL divergence, and classification loss.
   - **Training**: The model is trained using a custom training loop to optimize the loss function.

3. **Latent Representation and Classification:**
   - **Latent Representations**: Obtains latent representations from the trained encoder.
   - **Classification**: Performs binary classification on the latent space representations and evaluates accuracy.

4. **Dimensionality Reduction and Feature Importance Analysis:**
   - **PCA**: Apply PCA for dimensionality reduction and use the elbow method to determine the optimal number of principal components.
   - **SHAP Analysis**:
     - **Model Training**: Train a Random Forest model on the top PCA components to predict a subset of original features.
     - **SHAP Values**: Calculate SHAP values to explain the impact of each PCA component on the model’s predictions.
     - **Visualization**: Generate SHAP summary and dependence plots to visualize feature importance.

## Requirements

- Python (>=3.6)
- NumPy
- TensorFlow
- Keras
- Matplotlib
- Scikit-learn
- SHAP

## Usage

1. **Install the Required Libraries**: Ensure all necessary libraries installed.
2. **Run the Provided Python Code**: Execute the Python code in an environment such as Google Colab or a local setup.
3. **Adjust Parameters**: Modify parameters such as the number of epochs, batch size, and latent dimension as needed for your specific use case.

## Notes

- **Synthetic Data**: The provided code uses some fake simplified synthetic data. Replace it with real multi-omics preprocessed (ready-to-encode) data for practical applications.
- **Data Sparsity**: When using real single-cell data, consider regularizing sparsity (e.g., using dropout) and exploring alternative architectures such as Convolutional Neural Networks (CNNs) for imaging data.
- **Model Exploration**: Experiment with different VAE architectures and hyperparameters for optimal performance (e.g., using different loss functions).
- **SHAP Analysis**: SHAP can be applied to different subsets of features or machine learning models to enhance interpretability and explainable AI.
