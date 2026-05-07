import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
import os

def visualize_dtw_distance_matrix(distance_matrix, output_path="heatmap_output.png"):
    """
    Plots the N×M distance matrix as a heatmap.
    A successful alignment should show a clear, dark, diagonal 'staircase' representing the D3TW path.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(distance_matrix, cmap="viridis", annot=False)
    plt.title("D3TW Alignment Distance Matrix")
    plt.xlabel("English Translations / Target Patches")
    plt.ylabel("Arabic Visual Patches")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved D3TW Heatmap to {output_path}")
    plt.close()

def map_bounding_boxes(arabic_img, english_img, path, output_path="bb_mapping_output.png"):
    """
    Takes an Arabic image and an English image and physically draws lines connecting the 
    corresponding ink patches based on the D3TW alignment path.
    
    path: Array of coordinate tuples [(x1,y1), (x2,y2)...] of corresponding indices
    """
    # Assuming horizontal concatenation of the images with a space between them
    h1, w1 = arabic_img.shape[:2]
    h2, w2 = english_img.shape[:2]
    
    max_h = max(h1, h2)
    total_w = w1 + w2 + 100 # Add a 100px padding between them
    
    canvas = np.ones((max_h, total_w, 3), dtype=np.uint8) * 255
    canvas[:h1, :w1] = arabic_img
    canvas[:h2, w1+100:w1+100+w2] = english_img
    
    # Calculate mapping coordinates: Assuming patching was linear from left to right or top to bottom.
    # Since Arabic goes Right to Left and English Left to Right, handle accordingly depending on patch extraction algorithm.
    for (arabic_idx, english_idx) in path:
        # Example: just dummy coordinates. Replace with bounding box center points extraction.
        ar_cx = min(int((arabic_idx / max(1, path[-1][0])) * w1), w1 - 1)
        en_cx = w1 + 100 + min(int((english_idx / max(1, path[-1][1])) * w2), w2 - 1)
        
        ar_cy = h1 // 2
        en_cy = h2 // 2
        
        cv2.line(canvas, (ar_cx, ar_cy), (en_cx, en_cy), (0, 0, 255), 2)
        
    cv2.imwrite(output_path, canvas)
    print(f"Saved Visual Bounding Box Mapping to {output_path}")

if __name__ == "__main__":
    print("Executing Phase 3: Qualitative Visualization (The Eye Test)")
    # Generate dummy data for illustration
    dist_matrix = np.random.rand(50, 50) * 10
    for i in range(50):
        for j in range(-3, 3):
            if 0 <= i + j < 50:
                dist_matrix[i, i+j] = dist_matrix[i, i+j] * 0.1 # Dark diagonal
    
    visualize_dtw_distance_matrix(dist_matrix, "distance_matrix_sample.png")
    
    # Create simple dummy images
    arabic = np.ones((100, 400, 3), dtype=np.uint8) * 200
    english = np.ones((100, 400, 3), dtype=np.uint8) * 200
    dummy_path = [(i, i) for i in range(50)]
    map_bounding_boxes(arabic, english, dummy_path, "visual_mapping_sample.png")
