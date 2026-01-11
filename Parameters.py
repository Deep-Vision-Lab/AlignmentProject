import torch

# Model parameters
loss_type = 'SoftCrossEntropy'  # ['HeightDiff', 'MSE', 'GuidedAttention', 'KL-Divergence', 
                                #'Dice', 'Wasserstein', 'SoftCrossEntropy']
model_arch = 'CNN' # ['CNN-Transformer', 'CNN', 'dinov2', 'Transformer']
normalize_type = 'average' # ['min_max', 'mean_std', 'average']
device = "cuda" if torch.cuda.is_available() else "cpu"

# Training parameters
batch_size = 4
epochs = 300
learning_rate = 1e-4

# Data parameters
window_size = 16
vector_size = 128
lang = 'Arabic' # ['English', 'Arabic']

# Debugging and visualization parameters
debug = True # Set to True to save patches and heatmaps for debugging
debug_wandb = True # Set to True to log training to Weights & Biases
show_gradients = False # Set to True to print gradients for debugging
preprocess = 'none' # Set to True to normalize before loss computation
                    # ['none', 'Normalize', 'ExtractPaths']
Regular_ScoreMatrix_Load = False  # Set to True to load regular score matrices, False to load diff NW matrices

# Needleman-Wunsch parameters
matchScore = 1
mismatchScore = -3
gapScore = -1
