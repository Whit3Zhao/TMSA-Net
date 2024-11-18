# **TMSA-Net: A Novel Attention Mechanism for Improved Motor Imagery EEG Signal Processing**

This repository provides the implementation of **TMSA-Net**, a transformer-based deep learning model designed to enhance motor imagery EEG signal processing. The model incorporates a novel attention mechanism to effectively extract and utilize both spatial and temporal features in EEG data.

---

## **Features**
- **Novel Attention Mechanism**: Integrates both local and global attention modules to enhance feature representation.
- **Transformer-Based Architecture**: Leverages transformer modules to model temporal dependencies in EEG signals.
- **Dataset Compatibility**: Customizable for popular EEG datasets like BCIC-IV-2a, BCIC-IV-2b, and HGD.
- **Comprehensive Pipeline**: End-to-end implementation covering data preprocessing, training, evaluation, and visualization.
- **Explainable AI**: Includes Grad-CAM for interpreting model decisions.

---

## **Project Structure**
```plaintext
TMSA-Net/
├── LICENSE               # License file
├── README.md             # Project documentation
├── config.py             # Global configuration file for the project
├── main.py               # Entry script for training and evaluation
├── train.py              # Training and evaluation logic
├── process_hgd.py        # Script for preprocessing the HGD dataset
├── eeg_dataset.py        # EEG dataset class and data loading functions
├── util.py               # Grad-CAM and data augmentation tools
├── network/              # Model architecture implementations
│   └── TMSANet.py        # Implementation of TMSA-Net
├── output/               # Directory for saving outputs (models, logs, etc.)
```
---

## File Descriptions

### 1. `main.py`
The primary script for managing the training and evaluation pipeline. Key functionalities:
- Configures the environment (random seeds, GPU, etc.).
- Prepares the datasets for training and testing.
- Initializes the TMSA-Net model and performs per-subject training.
- Logs metrics (accuracy and Cohen's kappa) and saves the best model.

### 2. `train.py`
Implements the training and evaluation logic:
- **Training loop:** Computes loss, updates weights, and tracks metrics.
- **Evaluation loop:** Assesses model performance on the test set.
- **Model checkpointing:** Saves the best-performing model and generates loss/accuracy plots.

### 3. `config.py`
A centralized file for managing global configurations, including:
- Dataset paths and filenames.
- Model hyperparameters (e.g., embedding dimensions, attention heads).
- Training parameters (e.g., batch size, epochs, learning rate).

### 4. `process_hgd.py`
Preprocesses the HGD dataset for compatibility with TMSA-Net.

### 5. `eeg_dataset.py`
Defines the `eegDataset` class and related data loading functions:
- Handles `.mat` file loading and preprocessing.
- Supports reshaping, normalizing, and shuffling EEG data.

### 6. `util.py`
Provides utility functions, including:
- **Data Augmentation:** Generates augmented EEG data using segment-based techniques.
- **Grad-CAM:** Visualizes model attention to improve interpretability.

### 7. `network/TMSANet.py`
The core implementation of TMSA-Net, comprising:
- **Feature Extraction Module:** Captures spatial and temporal features from EEG signals.
- **Transformer Module:** Learns temporal dependencies with multi-head attention.
- **Classification Module:** Outputs class probabilities for motor imagery tasks.

---

## Supported Datasets
TMSA-Net supports multiple EEG datasets. Below are the recommended settings:

### 1. BCIC-IV-2a
- **Model Initialization:** `TMSANet(22, 1, 1000, 4)`
- **Key Parameters:**
  - `embed_dim = 19`
  - `num_classes = 4`

### 2. BCIC-IV-2b
- **Model Initialization:** `TMSANet(3, 1, 1000, 2)`
- **Key Parameters:**
  - `embed_dim = 6`
  - `num_classes = 2`

### 3. HGD
- **Model Initialization:** `TMSANet(44, 1, 1125, 4, embed_dim=10, attn_drop=0.7)`
- **Key Parameters:**
  - `embed_dim = 10`
  - `num_classes = 4`
  - `attn_drop = 0.7`

---

## Usage

### 1. Install Dependencies
Install the required Python packages using:
```bash
pip install -r requirements.txt
```
### 2.  Prepare Dataset
Organize the EEG dataset in the following structure:
```plaintext
data_path/
├── subject1/
│   ├── training.mat
│   ├── evaluation.mat
├── subject2/
│   ├── training.mat
│   ├── evaluation.mat
```
Update `config.py` with the appropriate dataset path:
```python
data_path = 'E:/EEG/dataset/newbcicIV2a/'
train_files = ['training.mat']
test_files = ['evaluation.mat']
```
### 3.  Train the Model
Run the training script:
```bash
python main.py
```

### 4.  Outputs
- Models are saved in the `output/` directory.
- Training logs and performance metrics are also saved.
- Visualization plots (e.g., `training_plots.png`) are generated.

---

## Results
After training, the following metrics will be reported for each subject:
-**Training and testing accuracy**
-**Cohen's kappa**
-**Average accuracy across all subjects**
## Example Output
```plaintext
------start Subject A01 training------
16,430,336 training parameters.
Epoch [1] | Train Loss: 0.543212  Train Accuracy: 0.752341 | Test Loss: 0.432112  Test Accuracy: 0.832451 | lr: 0.001000
...
subject:A01,accuracy:0.832451,kappa:0.751234
subject:A02,accuracy:0.845321,kappa:0.762341
average accuracy: 0.838886  average kappa: 0.756788
```

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## Citation
If you use this repository or the TMSA-Net model in your research, please cite:
```plaintext

```

---

## Contact
For questions or issues, please open an issue or contact:
<a href="mailto:zhaoqian0120@qq.com">📧 zhaoqian0120@qq.com</a>
