# Multi-Omics Data Integration with Variational Autoencoders (VAE) and SHAP Analysis

This project demonstrates how to integrate multi-omics data using a Variational Autoencoder (VAE) and subsequently perform dimensionality reduction and feature importance analysis using Principal Component Analysis (PCA) and SHAP (SHapley Additive exPlanations).

## Steps

1. **Data Generation and Preprocessing:**
   - Synthetic multi-omics data is generated (scRNA-seq, scATAC-seq, cell painting).
   - Data is combined and normalized.

2. **VAE Model:**
   - A VAE model is defined with an encoder and decoder to learn a latent representation of the data.
   - The model is trained using a custom loss function (reconstruction loss + KL divergence).

3. **Latent Representation:**
   - Latent representations of the data are obtained using the trained encoder.

4. **PCA and Elbow Method:**
   - PCA is applied to the combined data for dimensionality reduction.
   - The elbow method is used to determine the optimal number of principal components.

5. **SHAP Analysis:**
   - A Random Forest model is trained on the top PCA components to predict a subset of original features.
   - SHAP values are calculated to explain the impact of each PCA component on the model's predictions.
   - SHAP summary and dependence plots are generated to visualize feature importance.

## Requirements

- Python (>=3.6)
- NumPy
- Pandas
- TensorFlow
- Keras
- Matplotlib
- Scikit-learn
- SHAP

## Usage

1. Install the required libraries.
2. Run the provided Python code in a Google Colab environment.
3. Adjust parameters (e.g., number of epochs, batch size, number of PCA components) as needed.

## Notes

- The provided code uses synthetic data. Replace it with your own multi-omics datasets.
- If taking preprocessed data from real single cell experiment consider regularizing sparsity (e.g. NN dropout) instead of dense NN layers and also consider using CNN for imaging data
- Consider exploring different VAE architectures and hyperparameters for optimal performance (e.g. BCE as Loss).
- SHAP analysis can be applied to different subsets of features or different machine learning models just to contribute to the notion of explainable DL.
