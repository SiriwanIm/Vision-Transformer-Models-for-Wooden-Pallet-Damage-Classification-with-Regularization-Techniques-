# Vision Transformer Models for Wooden Pallet Damage Classification with Regularization Techniques

## Overview
This project focuses on classifying wooden pallet damage using Vision Transformer (ViT) models to enhance quality control in logistics and warehouse operations. Traditional manual inspections are prone to errors and inefficiencies. This solution leverages deep learning to automate the detection of defects. The study explores various regularization techniques, including batch normalization and dropout, to improve classification performance. The dataset consists of images of good and damaged wooden pallets collected from multiple sources. 

## Project Motivation
In manufacturing and logistics, pallet integrity is crucial for:
-	Ensuring product safety
-	Preventing workplace injuries
-	Reducing operational costs
-	Improving inventory management

## Features
-	**Transfer Learning:** Utilized pre-trained Vision Transformer models (DeiT-Tiny, ViT-L/16, ViT-B/16).
-	**Custom Classifier Head:** Designed a fully connected layer with ReLU activation.
-	**Regularization Strategies:**  Applied batch normalization and dropout to evaluate their impact on classification accuracy.
-	**Preprocessing:** Random cropping (224x224),and rescaling.
-	**Data Augmentation & Oversampling:** Addressed class imbalance using augmentation techniques and oversampling.

## Dataset
The dataset consists of images of good and damaged wooden pallets (2 classes):
-	**Good Pallets:**  358 images
-	**Damaged Pallets:** 824 images
-	**Total:** 1,182 images
From the dataset, each class was divided into a train set, a validation set, and a test set, respectively, at a ratio of 70%, 15%, and 15%.

##### Example of normal pallet.
![image](https://github.com/user-attachments/assets/e26ab82d-3250-42d6-8f80-83933ef39ce1)

##### Example of damaged pallet.
![image](https://github.com/user-attachments/assets/9f6d8d70-b303-495b-a5b2-64bff080f01a)


#### Note on Data Access
The data supporting the findings of this study are not publicly accessible due to company policy.

## Model Architecture
1. **Feature Extractor:** Pre-trained Vision Transformer (ViT-B/16, DeiT-Tiny, ViT-L/16)
2. **Classifier Head:**
- Dense (512) → ReLU → Batch Normalization → Dropout (50% and 25% )
- Dense (256) → ReLU → Batch Normalization → Dropout (50% and 25%)
- Dense (2) → Softmax
3. **Training Setup:**
- Loss function: Binary Cross-Entropy
- Optimizer: Adam
- Batch size: 32
- Learning rate: 0.001
- Training duration: 50 epochs

<img width="499" alt="Capture" src="https://github.com/user-attachments/assets/e3a403d5-7649-42a6-ae47-e1cf8b278e19" />

## Experiment
The experiment was designed by dividing it into 4 experiments: 1) Initial architecture, 2) Regularization with Batch Normalization 3) Regularization with Drop Out. 4) Regularization with coupling of Batch Normalization and Drop out. The illustration of classifier head of each experiment as shown in Table . The performance of these 4 experiments was compared across three ViT models to identify the best proposed architecture.

##### Classifier head of each experiment.
| Layer (Type)               | Initial Architecture | Batch Normalization | Dropout            | BatchNorm + Dropout |
|----------------------------|----------------------|---------------------|--------------------|---------------------|
| Dense (512 nodes)          | ✓                    | ✓                   | ✓                  | ✓                   |
| Activation (ReLU)          | ✓                    | ✓                   | ✓                  | ✓                   |
| Batch Normalization        | ✗                    | ✓                   | ✗                  | ✓                   |
| Dropout                    | ✗                    | ✗                   | 50% or 25%         | 50% or 25%          |
| Dense (256 nodes)          | ✓                    | ✓                   | ✓                  | ✓                   |
| Activation (ReLU)          | ✓                    | ✓                   | ✓                  | ✓                   |
| Batch Normalization        | ✗                    | ✓                   | ✗                  | ✓                   |
| Dropout                    | ✗                    | ✗                   | 50% or 25%         | 50% or 25%          |
| Dense (2 nodes)            | ✓                    | ✓                   | ✓                  | ✓                   |
| Activation (Softmax)       | ✓                    | ✓                   | ✓                  | ✓                   |

## Results and Conclusion
##### Model Performance Comparison (% Accuracy)

| Model        | Initial   | Batch Norm | Dropout (50%) | Dropout (25%) | Batch Norm + Dropout (25%) |
|--------------|-----------|------------|---------------|---------------|----------------------|
| DeiT-Tiny    | 98.31%    | 98.12%     | 97.93%        | 98.87%        | 97.55%               |
| ViT-L/16     | 96.61%    | 94.73%     | 94.73%        | 96.42%        | 93.79%               |
| ViT-B/16     | 98.31%    | 97.74%     | 97.18%        | 98.49%        | **99.06%**           |


Key observations to highlight:
1.	**Best Performer:** ViT-B/16 with BatchNorm + Dropout (25%) achieved highest accuracy (99.06%)
2.	**Dropout Impact:** 25% dropout generally outperformed 50% across all models
3.	**Architecture Sensitivity:** ViT-L/16 showed most variance with regularization
ViT-B/16 demonstrated the most robustness to regularization, with the combined BatchNorm + Dropout (25%) configuration yielding optimal results. Notably, initial architectures DeiT-Tiny maintained strong performance without complex regularization, suggesting potential efficiency advantages for resource-constrained deployments.




