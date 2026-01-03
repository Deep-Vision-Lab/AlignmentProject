# english_image_generator.py
# Description: Generates pairs of images with English text and their corresponding
#              Smith-Waterman alignment scoring matrices.
# Dependencies: Pillow, tqdm, numpy
# To install: pip install Pillow tqdm numpy

from PIL import Image, ImageDraw, ImageFont
import random
import os
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
from Parameters import *
from DiffNWAlgo import DiffNWAlgo
import warnings
warnings.filterwarnings("ignore")

# --- Needleman-Wunsch Algorithm Implementation ---
def needleman_wunsch_matrix(seq1, seq2, match_score=2, mismatch_penalty=-1, gap_penalty=-1):
    rows = len(seq1) + 1
    cols = len(seq2) + 1
    score_matrix = np.zeros((rows, cols), dtype=int)
    for i in range(rows):
        score_matrix[i, 0] = i * gap_penalty
    for j in range(cols):
        score_matrix[0, j] = j * gap_penalty
    for i in range(1, rows):
        for j in range(1, cols):
            similarity = match_score if seq1[i - 1] == seq2[j - 1] else mismatch_penalty
            diagonal_score = score_matrix[i - 1, j - 1] + similarity
            up_score = score_matrix[i - 1, j] + gap_penalty
            left_score = score_matrix[i, j - 1] + gap_penalty
            score_matrix[i, j] = max(diagonal_score, up_score, left_score)
    diff_NW_matrix = score_matrix[1:, 1:]
    return diff_NW_matrix

def save_matrix_to_file(matrix, filepath):
    try:
        np.save(filepath, matrix)
    except Exception as e:
        print(f"Error saving matrix to {filepath}: {e}")

def create_similarity_matrix(seq1, seq2):
    rows = len(seq1)
    cols = len(seq2)
    similarity_matrix = np.zeros((rows, cols), dtype=int)
    for i in range(rows):
        for j in range(cols):
            if seq1[i] == seq2[j]:
                similarity_matrix[i, j] = 1
    return similarity_matrix

def create_english_text_image(text_to_render, font_path_or_name, font_size_px, image_dimensions_px, output_image_path,
                             text_color="white", background_color="black", padding=20):
    try:
        font = ImageFont.truetype(font_path_or_name, font_size_px)
    except IOError:
        print(f"Critical: Fallback fonts not found for {output_image_path}. Using PIL's default font.")
        font = ImageFont.load_default()
    image = Image.new("RGB", (1000, 1000), background_color)
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = draw.textbbox((0, 0), text_to_render, font=font)
    text_width = right - left
    text_height = bottom - top
    image_width = text_width + padding * 2
    image_height = text_height + padding * 2
    image = Image.new('RGB', (image_width, image_height), color=(0, 0, 0))
    draw = ImageDraw.Draw(image)
    position = (padding, padding)
    draw.text(position, text_to_render, (255, 255, 255), font=font)
    image = image.resize(image_dimensions_px)
    image.save(output_image_path)

def save_text_to_file(text, filepath):
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
    except Exception as e:
        print(f"Error saving text to {filepath}: {e}")

if __name__ == "__main__":
    num_samples_to_generate = 3000
    base_output_directory = "DataSet/Synthetic_English"
    EMPTY_AUGMENT_PROBABILITY = 0.2
    output_images_dir = os.path.join(base_output_directory, "images")
    output_matrices_dir = os.path.join(base_output_directory, "matrices")
    output_similarity_matrices_dir = os.path.join(base_output_directory, "similarity_matrices")
    output_diff_nw_matrices_dir = os.path.join(base_output_directory, "diff_nw_matrices")
    output_text_lines_dir = os.path.join(base_output_directory, "texts")
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_matrices_dir, exist_ok=True)
    os.makedirs(output_similarity_matrices_dir, exist_ok=True)
    os.makedirs(output_diff_nw_matrices_dir, exist_ok=True)
    os.makedirs(output_text_lines_dir, exist_ok=True)
    print(f"Output will be saved in: {os.path.abspath(base_output_directory)}")
    print(f"Images will be in: {os.path.abspath(output_images_dir)}")
    print(f"Matrices will be in: {os.path.abspath(output_matrices_dir)}")
    print(f"Similarity matrices will be in: {os.path.abspath(output_similarity_matrices_dir)}")
    print(f"Diff NW matrices will be in: {os.path.abspath(output_diff_nw_matrices_dir)}")
    print(f"Text lines will be in: {os.path.abspath(output_text_lines_dir)}")
    diff_nw_algo = DiffNWAlgo(match_score=matchScore, miss_score=mismatchScore, gap=gapScore, batch=False)
    base_english_phrases = [
        "The sun shines",
        "Morning coffee helps",
        "The park is colorful",
        "Reading feeds the mind",
        "Travel opens minds",
        "Sports keep healthy",
        "Music calms nerves",
        "Work pays off",
        "Tech helps life",
        "Teamwork wins",
        "Flowers bloom",
        "The sea is calm",
        "History teaches",
        "Nature needs care",
        "Hope gives strength",
        "Stories are fun",
        "Leaves fall",
        "Family supports",
        "A smile spreads joy",
        "Stars shine bright",
        "Dreams need effort",
        "Seasons change",
        "Food is tasty",
        "Listening matters",
        "Friendship is treasure",
        "Weather is sunny",
        "New book excites",
        "It is calm now",
        "Good news came",
        "Time flies",
    ]
    augmenting_english_phrases = [
        "Surely",
        "Sometimes",
        "Generally",
        "Likely",
        "Exactly",
        "For this reason",
        "Sadly",
        "Luckily",
        "Soon",
        "Recently",
        "So far",
        "Early evening",
        "Before noon",
        "All day",
        "Slowly",
        "Quickly",
        "No doubt",
        "Never",
        "Again",
        "Here",
        "Another reason",
        "For example",
        "Above all",
        "Anytime",
        "By the window",
        "Truly",
        "Often",
        "Maybe",
        "Definitely",
        "Today",
    ]
    font_to_use = "Fonts/DejaVuSans.ttf"  # Ensure this path is correct or use a system font
    text_font_size = 60
    img_width = 1024
    img_height = 128
    image_dimensions = (img_width, img_height)
    NW_match_score = matchScore
    NW_mismatch_penalty = mismatchScore
    NW_gap_penalty = gapScore
    print(f"\nStarting generation of {num_samples_to_generate} sample pairs (image + matrix + text lines)...")
    def get_augment_phrase(phrases_list, empty_prob):
        if not phrases_list:
            return ""
        if random.random() < empty_prob:
            return ""
        return random.choice(phrases_list)
    for i in tqdm(range(1, num_samples_to_generate + 1), desc="Generating Samples"):
        if len(base_english_phrases) >= 2:
            common_phrases = random.sample(base_english_phrases, 2)
            common_phrase1 = common_phrases[0]
            common_phrase2 = common_phrases[1]
        elif len(base_english_phrases) == 1:
            common_phrase1 = base_english_phrases[0]
            common_phrase2 = base_english_phrases[0]
        else:
            print("Warning: Not enough base phrases. Using placeholders.")
            common_phrase1 = "Base phrase one"
            common_phrase2 = "Base phrase two"
        aug1_prefix = get_augment_phrase(augmenting_english_phrases, EMPTY_AUGMENT_PROBABILITY)
        aug1_middle = get_augment_phrase(augmenting_english_phrases, EMPTY_AUGMENT_PROBABILITY)
        aug1_suffix = get_augment_phrase(augmenting_english_phrases, EMPTY_AUGMENT_PROBABILITY)
        aug2_prefix = get_augment_phrase(augmenting_english_phrases, EMPTY_AUGMENT_PROBABILITY)
        aug2_middle = get_augment_phrase(augmenting_english_phrases, EMPTY_AUGMENT_PROBABILITY)
        aug2_suffix = get_augment_phrase(augmenting_english_phrases, EMPTY_AUGMENT_PROBABILITY)
        sentence1_parts = [aug1_prefix, common_phrase1, aug1_middle, common_phrase2, aug1_suffix]
        sentence2_parts = [aug2_prefix, common_phrase1, aug2_middle, common_phrase2, aug2_suffix]
        english_sentence_1 = " ".join(part.strip() for part in sentence1_parts if part.strip())
        english_sentence_2 = " ".join(part.strip() for part in sentence2_parts if part.strip())
        if not english_sentence_1: english_sentence_1 = common_phrase1
        if not english_sentence_2: english_sentence_2 = common_phrase2
        if random.choice([True, False]):
            english_sentence_1, english_sentence_2 = english_sentence_2, english_sentence_1
        output_img_file_1 = os.path.join(output_images_dir, f"img1_{i}.png")
        output_img_file_2 = os.path.join(output_images_dir, f"img2_{i}.png")
        output_matrix_file = os.path.join(output_matrices_dir, f"scoreMatrix_{i}.npy")
        output_similarity_matrix_file = os.path.join(output_similarity_matrices_dir, f"similarityMatrix_{i}.npy")
        output_diff_nw_matrix_file = os.path.join(output_diff_nw_matrices_dir, f"diffNWMatrix_{i}.npy")
        output_text_file_1 = os.path.join(output_text_lines_dir, f"text1_{i}.txt")
        output_text_file_2 = os.path.join(output_text_lines_dir, f"text2_{i}.txt")
        create_english_text_image(english_sentence_1, font_to_use, text_font_size, image_dimensions, output_img_file_1)
        create_english_text_image(english_sentence_2, font_to_use, text_font_size, image_dimensions, output_img_file_2)
        seq1_no_spaces = english_sentence_1.replace(" ", "")
        seq2_no_spaces = english_sentence_2.replace(" ", "")
        scoring_matrix = needleman_wunsch_matrix(
            seq1_no_spaces,
            seq2_no_spaces,
            match_score=NW_match_score,
            mismatch_penalty=NW_mismatch_penalty,
            gap_penalty=NW_gap_penalty
        )
        similarity_matrix = create_similarity_matrix(seq1_no_spaces, seq2_no_spaces)
        save_matrix_to_file(similarity_matrix, output_similarity_matrix_file)
        sim_tensor = torch.tensor(similarity_matrix, dtype=torch.float32)
        scaled_sim = sim_tensor * (matchScore - mismatchScore) + mismatchScore
        scaled_sim_batch = scaled_sim
        with torch.no_grad():
            diff_nw_align = diff_nw_algo(similarity_matrix=scaled_sim_batch, calc_cosine=False)
        diff_nw_matrix = diff_nw_align.squeeze(0).cpu().numpy()
        save_matrix_to_file(diff_nw_matrix, output_diff_nw_matrix_file)
        fig, axes = plt.subplots(1, 1, figsize=(20, 10))
        sns.heatmap(scoring_matrix, cmap='jet', linewidths=0.1, linecolor='black',
                    ax=axes, yticklabels=list(seq1_no_spaces),
                    xticklabels=list(seq2_no_spaces))
        save_matrix_to_file(scoring_matrix, output_matrix_file)
        plt.close(fig)
        save_text_to_file(english_sentence_1, output_text_file_1)
        save_text_to_file(english_sentence_2, output_text_file_2)
    print(
        f"\nScript finished. {num_samples_to_generate * 2} images, {num_samples_to_generate} matrices (as .npy), and {num_samples_to_generate * 2} text lines generated in '{base_output_directory}'.")
