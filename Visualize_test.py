from newDataLoader import train_dataloader
from Parameters import window_size, vector_size
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
from embeddingModel import sliding_window

if __name__ == '__main__':
    # Test the DataLoader
    for batch in train_dataloader:
        images_a, images_b, NW_matrices, similar_matrix, images1_names, images2_names = batch
        print("Images A shape:", images_a.shape)
        print("Images B shape:", images_b.shape)
        print("NW Matrices shape:", NW_matrices.shape)
        print("Similar Matrices shape:", similar_matrix.shape)
        print("Image 1 Names:", images1_names)
        print("Image 2 Names:", images2_names)
        break  # Just test one batch
    
    images_a = images_a.squeeze().detach().cpu().numpy()
    plt.imshow(images_a[0, 0, :, :], cmap='gray')
    plt.title("Sample Image A before Window Sliding")
    plt.axis('off')
    plt.savefig('full_sample_image_a.png')
    # Apply window sliding to images_a
    window_size = 16
    images_a = torch.tensor(sliding_window(torch.tensor(images_a), window_size, window_size//2))
    images_a = torch.flip(images_a, dims=[1])
    images_a = images_a.detach().cpu().numpy()
    # Visualize the first window (or change indices as needed)
    # Visualize the first channel/window as a 2D image
    num_windows = images_a.shape[1]
    fig, axes = plt.subplots(1, num_windows, figsize=(4 * num_windows, 4))
    for idx, i in enumerate(reversed(range(num_windows))):
        ax = axes[idx] if num_windows > 1 else axes
        ax.imshow(images_a[0, i, 0, :, :], cmap='gray')
        ax.set_title(f"Flipped Window {i}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('all_windows_image_a.png')
    plt.close()

    