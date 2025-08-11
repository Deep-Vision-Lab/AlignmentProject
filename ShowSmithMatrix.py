
import numpy as np
import matplotlib.pyplot as plt


from generateData import *

# matrix_path = "./DataSet/NewSynthetic/matrices/scoreMatrix_3000.npy"
# text1_path = "./DataSet/NewSynthetic/texts/text1_3000.txt"
# text2_path = "./DataSet/NewSynthetic/texts/text2_3000.txt"

# score_matrix = np.load(matrix_path)

# with open(text1_path, "r") as file:
#     text1 = file.read().strip()

# with open(text2_path, "r") as file:
#     text2 = file.read().strip()


def smith_waterman_matrix(seq1, seq2, match_score=2, mismatch_penalty=-1, gap_penalty=-1):
    """
    Computes the Smith-Waterman scoring matrix for local alignment.

    Args:
        seq1 (str): The first sequence (e.g., sentence).
        seq2 (str): The second sequence (e.g., sentence).
        match_score (int): Score for a match.
        mismatch_penalty (int): Penalty for a mismatch.
        gap_penalty (int): Penalty for a gap.

    Returns:
        numpy.ndarray: The scoring matrix (H-matrix) as a NumPy array.
    """
    rows = len(seq1) + 1
    cols = len(seq2) + 1

    # Initialize the scoring matrix with zeros
    score_matrix = np.zeros((rows, cols), dtype=int)  # Use NumPy array

    # Fill the scoring matrix
    for i in range(1, rows):
        for j in range(1, cols):
            # Score for match/mismatch between seq1[i-1] and seq2[j-1]
            similarity = match_score if seq1[i - 1] == seq2[j - 1] else mismatch_penalty

            diagonal_score = score_matrix[i - 1, j - 1] + similarity
            up_score = score_matrix[i - 1, j] + gap_penalty
            left_score = score_matrix[i, j - 1] + gap_penalty

            score_matrix[i, j] = max(0, diagonal_score, up_score, left_score)

    return score_matrix



text1 = "في بعض الأحيان الشمس مشرقة اليوم حقاً"
text2 = "في بعض الأحيان الشمس اليوم حقاً"
score_matrix = smith_waterman_matrix(text1, text2)

# Use characters as axis labels (or split into words if needed)
x_labels = list(text1)
y_labels = list(text2)

print(text1)
plt.figure(figsize=(max(8, len(x_labels) // 2), max(8, len(y_labels) // 2)))
plt.imshow(score_matrix, cmap='viridis', aspect='auto')
plt.colorbar(label='Score')
plt.xticks(ticks=np.arange(len(y_labels)), labels=y_labels, rotation=90, fontsize=8)
plt.yticks(ticks=np.arange(len(x_labels)), labels=x_labels, fontsize=8)
plt.xlabel('Text 2')
plt.ylabel('Text 1')
plt.title('Smith-Waterman Score Matrix Heatmap')

# Find the optimal local alignment path (traceback)
def smith_waterman_traceback(score_matrix, seq1, seq2):
    i, j = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
    path = []
    while score_matrix[i, j] > 0:
        path.append((i, j))
        diag = score_matrix[i-1, j-1] if i > 0 and j > 0 else -1
        up = score_matrix[i-1, j] if i > 0 else -1
        left = score_matrix[i, j-1] if j > 0 else -1
        max_score = max(diag, up, left)
        if max_score == diag:
            i -= 1
            j -= 1
        elif max_score == up:
            i -= 1
        else:
            j -= 1
    return path[::-1]  # reverse to get path from start to end

path = smith_waterman_traceback(score_matrix, text1, text2)

for (i, j), val in np.ndenumerate(score_matrix):
    plt.text(j, i, f"{val:.2f}", ha='center', va='center', color='white', fontsize=8)

# Plot the path on top of the heatmap
if path:
    path_x = [j for i, j in path]
    path_y = [i for i, j in path]
    plt.plot(path_x, path_y, color='red', linewidth=2, marker='o', markersize=5, label='Alignment Path')
    plt.legend()


# Generate 5 samples with two similar texts, two images, and one score matrix each
num_samples_to_generate = 5
base_output_directory = "DataSet/Trial"  # Main output directory

output_images_dir = os.path.join(base_output_directory, "images")
output_matrices_dir = os.path.join(base_output_directory, "matrices")
output_text_lines_dir = os.path.join(base_output_directory, "texts")

os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_matrices_dir, exist_ok=True)
os.makedirs(output_text_lines_dir, exist_ok=True)

font_to_use = "Fonts/Amiri-Regular.ttf"  # Ensure this path is correct
text_font_size = 90
img_width = 1024
img_height = 128
image_dimensions = (img_width, img_height)

base_texts = [
    "الشمس مشرقة اليوم حقاً",
    "الجو جميل في الخارج",
    "الطلاب يدرسون في المكتبة",
    "السيارة تسير بسرعة",
    "الطائر يطير في السماء"
]

for i in range(num_samples_to_generate):
    # Make two texts similar (identical or with minor change)
    text1 = base_texts[i]
    text2 = base_texts[i]  # identical for simplicity; you can add a small change if desired

    score_matrix = smith_waterman_matrix(text1, text2)

    output_img_file_1 = os.path.join(output_images_dir, f"img1_{i}.png")
    output_img_file_2 = os.path.join(output_images_dir, f"img2_{i}.png")
    output_matrix_file = os.path.join(output_matrices_dir, f"scoreMatrix_{i}.npy")
    output_text_file_1 = os.path.join(output_text_lines_dir, f"text1_{i}.txt")
    output_text_file_2 = os.path.join(output_text_lines_dir, f"text2_{i}.txt")

    create_arabic_text_image(text1, font_to_use, text_font_size, image_dimensions, output_img_file_1)
    create_arabic_text_image(text2, font_to_use, text_font_size, image_dimensions, output_img_file_2)
    save_matrix_to_file(score_matrix, output_matrix_file)
    save_text_to_file(text1, output_text_file_1)
    save_text_to_file(text2, output_text_file_2)