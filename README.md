# Machine Learning-Based Prediction of Large-for-Gestational Age Infants in Motherswith Gestational Diabetes Mellitus

Code, and trained models for "Machine Learning-Based Prediction of Large-for-Gestational Age Infants in Motherswith Gestational Diabetes Mellitus"

## Getting Started

### 1. Installation

- Clone this repo:

```bash
cd ~
git clone https://github.com/ERICSJTUER/LGAinGDM
cd LGAinGDM
```

### 2. Prerequisites

- Linux (Tested on Ubuntu 20.04)
- NVIDIA GPU (Tested on Nvidia GeForce RTX 3090)
- CUDA, CuDNN
- Python 3.7.12
- Pytorch>=0.12.0
- scikit-learn 1.0.2
- matplotlib = 3.1.1 
- numpy = 1.21.6
- pandas = 1.3.5

### 3. Code Base Structure

The code base structure is explained below:
- **main_fusion.py:** Script for training fusion model based on CGM and clinical data. 
- **main_clinical.py:** Script for training MLP-based model based on clinical data. 
- **main_CGM.py:** Script for training fusion model based on CGM  data. 
- **test.py:** You can use this script to test the model after training.

### 4. Training

 To train a model:

```bash
python main_fusion.py
```
- To see more intermediate results, check out  `./results/fusion`.

### 5. Testing

To test the model:

```bash
python test.py 
```

