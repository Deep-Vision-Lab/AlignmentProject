import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import numpy as np
import os
import argparse
from collections import Counter
import matplotlib.pyplot as plt

from embeddingModel import EmbeddingModel, sliding_window
from textEmbedding import build_text_embedder
from Parameters import window_size, stride_ratio, vector_size, device
from newDataLoader import build_dataloaders

def get_arabic_character_set(data_dir):
    """
    Scans the texts folder to dynamically build a clean, comprehensive set 
    of unique Arabic characters + space used in the actual dataset.
    """
    texts_dir = os.path.join(data_dir, "texts")
    if not os.path.exists(texts_dir):
        # Fallback to standard Arabic alphabet if texts directory doesn't exist
        print("Texts directory not found. Using default Arabic alphabet.")
        arabic_alphabet = "ابتثجحخدذرزسشصضطظعغفقكلمنهويءأإآىةؤئ "
        return sorted(list(set(arabic_alphabet)))
    
    char_counter = Counter()
    files = [f for f in os.listdir(texts_dir) if f.startswith("text1_") and f.endswith(".txt")]
    # Sample up to 1000 text files to find all active characters
    sampled_files = files[:1000]
    for file in sampled_files:
        with open(os.path.join(texts_dir, file), 'r', encoding='utf-8') as f:
            text = f.read().strip()
            char_counter.update(text)
            
    # Always ensure space is present
    char_counter[' '] = char_counter[' '] + 1
    
    # Sort the vocabulary of characters
    vocab = sorted(list(char_counter.keys()))
    print(f"Dynamically discovered {len(vocab)} unique characters in dataset.")
    return vocab

def run_clustering_and_letter_assignment(model_path, data_dir, sample_idx=0):
    # 1. Dynamically build character set of Arabic + Space
    vocab_chars = get_arabic_character_set(data_dir)
    vocab_size = len(vocab_chars)
    
    # 2. Setup Models
    stride = max(1, int(window_size * stride_ratio))
    model = EmbeddingModel(window_size=window_size, stride=stride, vector_size=vector_size, device=device)
    text_embedder = build_text_embedder(embedding_dim=vector_size)
    text_embedder = text_embedder.to(device)
    text_embedder.eval()
    
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        print("Model loaded successfully.")
    else:
        print("Warning: Model path not found, using random weights.")
    model.eval()

    # 3. Get Data sample
    train_dl, _, _ = build_dataloaders(data_dir)
    images, pos_texts, _ = next(iter(train_dl))
    image = images[sample_idx:sample_idx+1].to(device)
    text = pos_texts[sample_idx] # e.g. " كتاب "
    print(f"\nAnalyzing sample image index {sample_idx} with ground-truth text: '{text}'")

    # 4. Extract Visual and Text Embeddings
    with torch.no_grad():
        # Get visual embeddings for sliding window patches
        visual_features = model(image) # Shape: [1, seq_len, vector_size]
        visual_features = visual_features.squeeze(0) # Shape: [seq_len, vector_size]
        visual_norm = F.normalize(visual_features, p=2, dim=-1) # L2 Normalized [seq_len, vector_size]
        
        # Get character embeddings for all vocab characters (Arabic alphabet + space)
        # FastText or standard text_embedder
        vocab_embeddings = text_embedder(vocab_chars) # Shape: [vocab_size, 1, vector_size] or [vocab_size, vector_size]
        vocab_embeddings = vocab_embeddings.squeeze(1) if vocab_embeddings.dim() == 3 else vocab_embeddings
        vocab_norm = F.normalize(vocab_embeddings, p=2, dim=-1) # L2 Normalized [vocab_size, vector_size]

    # 5. Map Every Window to Its Closest Arabic Letter
    # We compute cosine similarity matrix between visual patches [seq_len, vector_size] 
    # and all vocabulary embeddings [vocab_size, vector_size]
    # Cosine similarity is dot product of normalized embeddings:
    # similarity: [seq_len, vocab_size]
    similarity = torch.mm(visual_norm, vocab_norm.transpose(0, 1))
    
    # Get the index of the highest similarity score for each patch window
    closest_vocab_indices = torch.argmax(similarity, dim=1).cpu().numpy()
    
    # Convert back to actual letters
    assigned_letters = [vocab_chars[idx] for idx in closest_vocab_indices]

    # 6. Perform K-Means Clustering on the Visual Patch Features
    # The number of clusters k matches the active character set size
    features_np = visual_features.cpu().numpy()
    seq_len = features_np.shape[0]
    num_clusters = min(vocab_size, seq_len)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(features_np)

    # 7. Print results showing correct clustering of duplicated letters
    print(f"\nClustering and Alphabet Assignment (Sequence Length = {seq_len}, Alphabet Size = {vocab_size}):")
    print(f"{'Patch Index':<12} | {'Cluster ID':<12} | {'Closest Alphabet Letter':<25} | {'Cosine Similarity Score':<25}")
    print("-" * 85)
    
    for i in range(seq_len):
        best_sim = similarity[i, closest_vocab_indices[i]].item()
        assigned_char = assigned_letters[i]
        # Label spaces clearly for readability
        char_label = f"SPACE (Index {closest_vocab_indices[i]})" if assigned_char == ' ' else f"'{assigned_char}' (Index {closest_vocab_indices[i]})"
        
        print(f"{i:<12} | {cluster_ids[i]:<12} | {char_label:<25} | {best_sim:.4f}")

    # Inspecting Duplicated Letters grouping
    print("\n--- Grouping Analysis (Check if duplicated letters fall into the same clusters) ---")
    cluster_to_chars = {}
    for i in range(seq_len):
        c_id = cluster_ids[i]
        char = assigned_letters[i]
        if c_id not in cluster_to_chars:
            cluster_to_chars[c_id] = []
        cluster_to_chars[c_id].append((i, char))
        
    for c_id in sorted(cluster_to_chars.keys()):
        members = cluster_to_chars[c_id]
        indices = [m[0] for m in members]
        chars = [f"'{m[1]}'" if m[1] != ' ' else "SPACE" for m in members]
        print(f"Cluster {c_id:<3}: Contains patches {str(indices):<18} aligned with characters: {', '.join(chars)}")

    # 8. Generate and Save Clustering Visualization Figure (TSNE Plot)
    print("\nGenerating TSNE visualization of visual clusters...")
    try:
        # Prevent headless display errors
        plt.switch_backend('Agg')
        
        # Use TSNE to project 128-d or 512-d embeddings to 2D
        # Set perplexity dynamically to prevent issues with small sequence lengths
        perplexity = min(30, max(5, seq_len // 3))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
        features_2d = tsne.fit_transform(features_np)
        
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=cluster_ids, cmap='tab20', s=100, alpha=0.8, edgecolors='none')
        
        # Annotate points with their closest Arabic letter
        for i in range(seq_len):
            char = assigned_letters[i]
            label = "SPACE" if char == ' ' else f"'{char}'"
            plt.annotate(f"{label} ({i})", (features_2d[i, 0], features_2d[i, 1]), fontsize=8, alpha=0.9, xytext=(5, 2), textcoords='offset points')
            
        plt.colorbar(scatter, label='Cluster ID')
        plt.title(f"TSNE Visualization of Sliding Window Clustering\nSample Text: '{text}'", fontsize=12, fontweight='bold', pad=15)
        plt.xlabel("TSNE Dimension 1")
        plt.ylabel("TSNE Dimension 2")
        plt.grid(True, linestyle='--', alpha=0.5)
        
        output_fig = "clustering_visualization.png"
        plt.savefig(output_fig, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Success! Figure saved to '{output_fig}'")
    except Exception as e:
        print(f"Failed to generate TSNE visualization: {e}")

    # 9. NEW: Generate and Save actual Image Patch Grid labeled with Clusters and Letters
    print("\nExtracting sliding window image patches and generating grid visualization...")
    try:
        # Extract sliding window patches
        # Shape: [1, num_windows, C, H, window_size]
        patches_tensor = sliding_window(image, window_size, stride)
        patches_tensor = patches_tensor.squeeze(0).cpu() # [num_windows, C, H, window_size]
        num_windows = patches_tensor.shape[0]
        
        # Calculate how many sequence features are generated per patch
        # Since seq_len = num_windows * seq_len_per_patch
        seq_len_per_patch = max(1, seq_len // num_windows)
        
        # Grid plot configurations
        cols = 8
        rows = int(np.ceil(num_windows / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 2 * rows))
        fig.suptitle(f"Visual Sliding Window Patches with Cluster IDs & Character Assignments\nText: '{text}'", fontsize=14, fontweight='bold', y=0.98)
        
        axes_flat = axes.flatten()
        
        for i in range(len(axes_flat)):
            ax = axes_flat[i]
            if i < num_windows:
                # Get the image patch [C, H, W]
                patch = patches_tensor[i]
                # Convert PyTorch tensor to [H, W, C] for matplotlib
                patch_np = patch.permute(1, 2, 0).numpy()
                # Denormalize if needed (assuming standard [0, 1] range)
                patch_np = np.clip(patch_np, 0.0, 1.0)
                
                ax.imshow(patch_np)
                
                # Retrieve the cluster and letter mapped to this window
                # We align with the corresponding feature index in the dense sequence
                feature_idx = min(i * seq_len_per_patch, seq_len - 1)
                char = assigned_letters[feature_idx]
                char_title = "SPACE" if char == ' ' else f"'{char}'"
                ax.set_title(f"W{i} | C{cluster_ids[feature_idx]}\n{char_title}", fontsize=8, fontweight='semibold')
            
            # Hide axes ticks and labels
            ax.axis('off')
            
        plt.tight_layout()
        output_grid_fig = "patch_images_clustering.png"
        plt.savefig(output_grid_fig, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Success! Visual patch grid saved to '{output_grid_fig}'")
    except Exception as e:
        print(f"Failed to generate visual patch grid: {e}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run custom OCR clustering analysis.')
    parser.add_argument('--model_path', type=str, default="Weights/job_1/model_latest.pth", help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default="DataSet/Synthetic_Arabic_10000", help='Dataset directory path')
    parser.add_argument('--sample_idx', type=int, default=1, help='Sample index in the dataloader')
    args = parser.parse_args()
    
    if os.path.exists(args.data_dir):
        run_clustering_and_letter_assignment(args.model_path, args.data_dir, args.sample_idx)
    else:
        print(f"Error: Dataset directory '{args.data_dir}' not found. Please provide a valid path using --data_dir.")
