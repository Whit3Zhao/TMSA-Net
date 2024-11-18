# ---- Data Settings ---- #
data_path = 'E:/EEG/dataset/newbcicIV2a/'  # Path to the EEG dataset
train_files = ['training.mat']            # Training dataset file name
test_files = ['evaluation.mat']           # Testing dataset file name
output = 'output'                         # Directory to save outputs (models, logs, etc.)
model_name = "test"                       # The name of model to save
batch_size = 16                           # Batch size for training and testing
num_segs = 8                              # Number of segments for data augmentation

# ---- Model Settings ---- #
pool_size = 50                            # Pooling kernel size
pool_stride = 15                          # Pooling stride size
num_heads = 4                             # Number of attention heads in the transformer
fc_ratio = 2                              # Feed-forward network expansion ratio
depth = 1                                 # Depth of the transformer encoder (number of layers)

# ---- Training Settings ---- #
epochs = 2000                             # Number of training epochs
lr = 2 ** -12                             # Learning rate
weight_decay = 1e-4                       # Weight decay for optimizer
