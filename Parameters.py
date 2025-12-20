import torch

loss_type = 'MSE' # ['HeightDiff', 'MSE', 'GuidedAttention', 'KL-Divergence', 'Dice', 'Wasserstein']
model_arch = 'CNN' # ['CNN-Transformer', 'CNN', 'Transformer']
window_size = 32
vector_size = 128
normalize_type = 'average' # ['min_max', 'mean_std', 'average']
epochs = 300
learning_rate = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
debug = True # Set to True to save patches and heatmaps for debugging
debug_wandb = False # Set to True to log training to Weights & Biases
show_gradients = True # Set to True to print gradients for debugging

matchScore = 10
mismatchScore = -27
gapScore = -10

batch_size = 4
