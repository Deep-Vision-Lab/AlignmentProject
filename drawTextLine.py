from PIL import Image, ImageDraw, ImageFont
import PIL
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from fetch_arabic_sentence import fetch_arabic_sentence
from splitTextLine import tokenize_based_on_non_connecting_letters
from augmentSenetece import augment_sentence
from matchingAlgorthim import smith_waterman
from matplotlib.patches import FancyArrow
import arabic_reshaper
from bidi.algorithm import get_display


# Step 1: Create an image with black background and white Arabic text

def create_text_image(text, font_path="arial.ttf", font_size=50, padding=10):
    """
       Create an image with black background and white Arabic text.

       :param text: The Arabic text to draw on the image.
       :param font_path: Path to the .ttf font file.
       :param font_size: Size of the font.
       :param padding: Padding around the text.
       :return: An image object with the drawn text.
       """
    # Load the font
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        return f"Font file not found: {font_path}"

    print(PIL.__version__)

    # Create a temporary image to get the size of the text
    temp_image = Image.new('RGB', (1000, 1000))  # Large enough temporary size
    draw = ImageDraw.Draw(temp_image)
    # Get the size of the text
    text_width, text_height = draw.textsize(text, font=font)

    # Calculate the size of the final image with some padding
    image_width = text_width + 2 * padding
    image_height = text_height + 2 * padding

    # Create the final image with the calculated size
    image = Image.new('RGB', (image_width, image_height), color=(0, 0, 0))
    draw = ImageDraw.Draw(image)
    # Calculate the position of the text to center it
    position = (padding, padding)

    # Add text to the image
    draw.text(position, text, (255, 255, 255), font=font)

    return image


def reshape_arabic_text_list(words):
    """Reshape a list of Arabic words for correct display."""
    reshaped_words = [get_display(arabic_reshaper.reshape(word)) for word in words]
    return reshaped_words


def plot_heatmap(matrix, seq1, seq2):
    """
    Plot a heatmap of the scoring matrix.

    :param matrix: The scoring matrix to plot.
    :param seq1: First sequence (list of subwords in Arabic).
    :param seq2: Second sequence (list of subwords in Arabic).
    """

    seq1 = reshape_arabic_text_list(seq1)
    seq2 = reshape_arabic_text_list(seq2)
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, cmap='viridis', annot=True, fmt=".2f",
                xticklabels=[''] + seq2, yticklabels=[''] + seq1)
    plt.title('Smith-Waterman Scoring Matrix Heatmap')
    plt.xlabel('Sequence 2')
    plt.ylabel('Sequence 1')
    plt.xticks(fontsize=12, )
    plt.yticks(fontsize=12)
    plt.show()


def plot_traceback(score_matrix, traceback_matrix, seq1, seq2):
    """Plot the scoring matrix with the traceback path."""

    plt.figure(figsize=(12, 10))
    seq1 = reshape_arabic_text_list(seq1)
    seq2 = reshape_arabic_text_list(seq2)
    sns.heatmap(score_matrix, cmap='viridis', annot=True, fmt=".2f",
                xticklabels=seq2, yticklabels=seq1, cbar_kws={'label': 'Score'})

    ax = plt.gca()

    # Set ticks in the middle of each cell
    ax.set_xticks(np.arange(len(seq2)) + 0.5)
    ax.set_yticks(np.arange(len(seq1)) + 0.5)
    ax.set_xticklabels(seq2, rotation=0, ha='center', fontsize=12, family='Amiri')
    ax.set_yticklabels(seq1, rotation=0, ha='center', fontsize=12, family='Amiri')

    # Traceback path
    i, j = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
    while i > 0 and j > 0 and score_matrix[i, j] != 0:
        if traceback_matrix[i, j] == 1:  # Diagonal
            ax.add_patch(FancyArrow(j + 0.5, i + 0.5, -1, -1, color='red', head_width=0.3, head_length=0.3))
            i, j = i - 1, j - 1
        elif traceback_matrix[i, j] == 2:  # Up
            ax.add_patch(FancyArrow(j + 0.5, i + 0.5, 0, -1, color='blue', head_width=0.3, head_length=0.3))
            i = i - 1
        elif traceback_matrix[i, j] == 3:  # Left
            ax.add_patch(FancyArrow(j + 0.5, i + 0.5, -1, 0, color='green', head_width=0.3, head_length=0.3))
            j = j - 1

    plt.title('Smith-Waterman Scoring Matrix Heatmap')
    plt.xlabel('Sequence 2')
    plt.ylabel('Sequence 1')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()


def create_text_image(text, font_path="arial.ttf", font_size=50, padding=10):
    """
    Create an image with black background and white Arabic text.
    """
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        return f"Font file not found: {font_path}"

    temp_image = Image.new('RGB', (1000, 1000))
    draw = ImageDraw.Draw(temp_image)
    text_width, text_height = draw.textsize(text, font=font)
    image_width = text_width + 2 * padding
    image_height = text_height + 2 * padding

    image = Image.new('RGB', (image_width, image_height), color=(0, 0, 0))
    draw = ImageDraw.Draw(image)
    position = (padding, padding)
    draw.text(position, text, (255, 255, 255), font=font)

    return image


def reshape_arabic_text_list(words):
    """Reshape a list of Arabic words for correct display."""
    reshaped_words = [get_display(arabic_reshaper.reshape(word)) for word in words]
    return reshaped_words


def plot_heatmap(matrix, seq1, seq2, ax):
    """Plot a heatmap of the scoring matrix."""
    seq1 = reshape_arabic_text_list(seq1)
    seq2 = reshape_arabic_text_list(seq2)
    sns.heatmap(matrix, cmap='viridis', annot=True, fmt=".2f", ax=ax, xticklabels=seq2, yticklabels=seq1,
                cbar_kws={'label': 'Score'})
    ax.set_title('Smith-Waterman Scoring Matrix Heatmap')
    ax.set_xlabel('Sequence 2')
    ax.set_ylabel('Sequence 1')


def plot_traceback(score_matrix, traceback_matrix, ax, seq1=[], seq2=[]):
    """Plot the scoring matrix with the traceback path."""
    if seq1 and seq2:
        seq1 = reshape_arabic_text_list(seq1)
        seq2 = reshape_arabic_text_list(seq2)
        seq1 = seq1[::-1]
        seq1 = seq1 + ['']*(score_matrix.shape[0] - len(seq1))
        seq1 = seq1[::-1]
        seq2 = seq2[::-1]
        seq2 = seq2 + ['']*(score_matrix.shape[1] - len(seq2))
        seq2 = seq2[::-1]
        sns.heatmap(score_matrix, cmap='viridis', annot=True, fmt=".2f", ax=ax,
                xticklabels=seq2, yticklabels=seq1, cbar=False)
    else:
        sns.heatmap(score_matrix, cmap='viridis', annot=True, fmt=".2f", ax=ax, cbar=False)

    # Traceback path
    i, j = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
    while i > 0 and j > 0 and score_matrix[i, j] != 0:
        if traceback_matrix[i, j] == 1:  # Diagonal
            ax.add_patch(FancyArrow(j + 0.5, i + 0.5, -1, -1, color='red', head_width=0.3, head_length=0.3))
            i, j = i - 1, j - 1
        elif traceback_matrix[i, j] == 2:  # Up
            ax.add_patch(FancyArrow(j + 0.5, i + 0.5, 0, -1, color='blue', head_width=0.3, head_length=0.3))
            i = i - 1
        elif traceback_matrix[i, j] == 3:  # Left
            ax.add_patch(FancyArrow(j + 0.5, i + 0.5, -1, 0, color='green', head_width=0.3, head_length=0.3))
            j = j - 1

    ax.set_title('Smith-Waterman Traceback')
    ax.set_xlabel('Sequence 2')
    ax.set_ylabel('Sequence 1')


def visualize_combined_image(text_image1, text_image2, score_matrix, traceback_matrix, seq1, seq2):
    """Combine the text line images with the traceback heatmap."""
    # Stack the text images vertically
    combined_image = Image.new('RGB', (max(text_image1.width, text_image2.width),
                                       text_image1.height + text_image2.height + 50), color=(255, 255, 255))
    combined_image.paste(text_image1, (0, 0))
    combined_image.paste(text_image2, (0, text_image1.height + 10))

    # Convert PIL image to numpy array for matplotlib
    combined_image_np = np.array(combined_image)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
    ax1.imshow(combined_image_np)
    ax1.axis('off')
    ax1.set_title('Original and Augmented Text Lines', fontsize=16)
    plot_traceback(score_matrix, traceback_matrix, seq1, seq2, ax2)
    plt.show()


# if __name__ == '__main__':
#     for i in range(0, 5):
#         _, sentencewithNwords = fetch_arabic_sentence(n=3)
#         subWordsOfNwords = tokenize_based_on_non_connecting_letters(sentencewithNwords)
#         _, word = fetch_arabic_sentence(n=1)
#         subWordsOfAddwords = tokenize_based_on_non_connecting_letters(word)
#         sentenceAddAugmetation = augment_sentence(subWordsOfNwords, new_words=subWordsOfAddwords, operation="add")
#
#         textLineOriginal = " ".join(subWordsOfNwords)
#         imgTextLineOriginal = create_text_image(textLineOriginal, font_path='fonts/Alexandria.ttf')
#         textLineAugmented = " ".join(sentenceAddAugmetation)
#         imgTextLineAugmented = create_text_image(textLineAugmented, font_path='fonts/Sahel.ttf')
#
#         print(imgTextLineOriginal)
#         plt.imshow(imgTextLineOriginal)
#         plt.imshow(imgTextLineAugmented)
#         plt.show()
#         score, alignment1, alignment2, score_matrix, track_matrix = smith_waterman(subWordsOfNwords, sentenceAddAugmetation)
#         print(  score, alignment1, alignment2, score_matrix, track_matrix)
#         visualize_combined_image(imgTextLineOriginal, imgTextLineAugmented, score_matrix, track_matrix, subWordsOfNwords, sentenceAddAugmetation)
#


if __name__ == '__main__':

    for i in range(0, 5):
        _, sentencewithNwords = fetch_arabic_sentence(n=3)
        subWordsOfNwords = tokenize_based_on_non_connecting_letters(sentencewithNwords)
        _, word = fetch_arabic_sentence(n=1)
        subWordsOfAddwords = tokenize_based_on_non_connecting_letters(word)
        print("Fetched Arabic Sentence:", subWordsOfNwords)
        print("Fetched Arabic Word:", subWordsOfAddwords)
        sentenceAddAugmetation = augment_sentence(subWordsOfNwords, new_words=subWordsOfAddwords, operation="add")
        print(" After Add Augmentation Arabic Sentence:", sentenceAddAugmetation)

        textLineOriginal = " ".join(subWordsOfNwords)
        imgTextLineOriginal = create_text_image(textLineOriginal, font_path='fonts/Alexandria.ttf')
        plt.imshow(imgTextLineOriginal)
        plt.show()
        textLineAugmented = " ".join(sentenceAddAugmetation)
        imgTextLineAugmented = create_text_image(textLineAugmented, font_path='fonts/Sahel.ttf')
        plt.imshow(imgTextLineAugmented)
        plt.show()

        score, alignment1, alignment2, score_matrix, track_matrix = smith_waterman(subWordsOfNwords,
                                                                                   sentenceAddAugmetation)
        print("Alignment Score:", score)
        print("Alignment 1:", alignment1)
        print("Alignment 2:", alignment2)

        fig, ax = plt.subplots(1, 1, figsize=(18, 10))
        # print(score_matrix)
        plot_heatmap(score_matrix, subWordsOfNwords, sentenceAddAugmetation, ax)
        plot_traceback(score_matrix, track_matrix, subWordsOfNwords, sentenceAddAugmetation, ax)
        plt.show()

# # Segment text into connected components
# segments = segment_text(text_image)
#
# # Draw bounding boxes on the image
# image_with_boxes = draw_bounding_boxes(text_image, segments)
#
# # Save images to visualize the result
# image_with_boxes.save("segmented_image.png")
#
# # Print each segment with its bounding box coordinates
# for idx, (segment_text, (x, y, w, h)) in enumerate(segments):
#     print(f"Segment {idx + 1}: '{segment_text.strip()}', Bounding Box: x={x}, y={y}, w={w}, h={h}")


# 1.function that create arabic senteces
# 1.1 write it with one font and apply blur.
# 2.function that split it components based on non-conncting letters and spaces and
#  2.1 also remove all not alphabtic letters ! such as "=,!@#%^&(...." (Optional Step )

# 3.from this sentece, remove word/letter/connected word,
# 4.check that after removing we have similiarty with threshould t (70% for example)
#  4.1 if not return to 4 till we have simlirity above this threshould
#  4.2 if yes write the second senetece with another font and apply blur
# 5. build  matrex of smith-watermelon of connect-componet level or letter level
# 6. convert this matrix to image ( such as thin heat map ) !!
# ALSO DRAW THE TRACK !!
# 7. we have now DATASET ready !
