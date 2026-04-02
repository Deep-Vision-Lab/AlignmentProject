# arabic_image_generator.py
# Description: Generates pairs of images with Arabic text and their corresponding
#              Smith-Waterman alignment scoring matrices.
# Dependencies: Pillow, arabic-reshaper, python-bidi, tqdm, numpy
# To install: pip install Pillow arabic-reshaper python-bidi tqdm numpy

import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont
import random
from Parameters import *
import os
import seaborn as sns

from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np  # Import numpy
import torch
from DiffNWAlgo import DiffNWAlgo

import warnings
warnings.filterwarnings("ignore")

# --- Needleman-Wunsch Algorithm Implementation ---
def needleman_wunsch_matrix(seq1, seq2, match_score=2, mismatch_penalty=-1, gap_penalty=-1):
    """
    Computes the Needleman-Wunsch scoring matrix for global alignment.
    Returns the matrix without the first row and column (removes empty string alignments).

    Args:
        seq1 (str): The first sequence (e.g., sentence).
        seq2 (str): The second sequence (e.g., sentence).
        match_score (int): Score for a match.
        mismatch_penalty (int): Penalty for a mismatch.
        gap_penalty (int): Penalty for a gap.

    Returns:
        numpy.ndarray: The scoring matrix as a NumPy array without first row/column.
    """
    rows = len(seq1) + 1
    cols = len(seq2) + 1

    # Initialize the scoring matrix
    score_matrix = np.zeros((rows, cols), dtype=int)

    # Initialize first row and column with gap penalties (global alignment)
    for i in range(rows):
        score_matrix[i, 0] = i * gap_penalty
    for j in range(cols):
        score_matrix[0, j] = j * gap_penalty

    # Fill the scoring matrix
    for i in range(1, rows):
        for j in range(1, cols):
            # Score for match/mismatch between seq1[i-1] and seq2[j-1]
            similarity = match_score if seq1[i - 1] == seq2[j - 1] else mismatch_penalty

            diagonal_score = score_matrix[i - 1, j - 1] + similarity
            up_score = score_matrix[i - 1, j] + gap_penalty
            left_score = score_matrix[i, j - 1] + gap_penalty

            score_matrix[i, j] = max(diagonal_score, up_score, left_score)

    # Remove first row and column to eliminate empty string alignments
    diff_NW_matrix = score_matrix[1:, 1:]
    
    return diff_NW_matrix



def save_matrix_to_file(matrix, filepath):
    """
    Saves a 2D matrix to a .npy file.

    Args:
        matrix (numpy.ndarray): The matrix to save.
        filepath (str): The path to the output .npy file.
    """
    try:
        np.save(filepath, matrix)
    except Exception as e:
        print(f"Error saving matrix to {filepath}: {e}")


def create_similarity_matrix(seq1, seq2):
    """
    Creates a binary similarity matrix where 1 indicates a match and 0 indicates a mismatch.

    Args:
        seq1 (str): The first sequence.
        seq2 (str): The second sequence.

    Returns:
        numpy.ndarray: A binary similarity matrix of shape (len(seq1), len(seq2)).
    """
    rows = len(seq1)
    cols = len(seq2)
    similarity_matrix = np.zeros((rows, cols), dtype=int)
    
    for i in range(rows):
        for j in range(cols):
            if seq1[i] == seq2[j]:
                similarity_matrix[i, j] = 1
    
    return similarity_matrix


# --- Image Generation Function (from previous version) ---
def create_arabic_text_image(text_to_render, font_path_or_name, font_size_px, image_dimensions_px, output_image_path,
                             text_color="white", background_color="black", padding=20):
    """
    Creates an image with the given Arabic text.
    (Handles reshaping, bidi, font loading, drawing, and saving)
    """
    reshaped_text = arabic_reshaper.reshape(text_to_render)
    bidi_text = get_display(reshaped_text)

    try:
        font = ImageFont.truetype(font_path_or_name, font_size_px)
    except IOError:
        print(f"Critical: Fallback fonts not found for {output_image_path}. Using PIL's default font.")

    image = Image.new("RGB", (1000, 1000), background_color)
    draw = ImageDraw.Draw(image)

    # Calculate the bounding box of the text
    left, top, right, bottom = draw.textbbox((0, 0), bidi_text, font=font)
    text_width = right - left
    text_height = bottom - top

    image_width = text_width + padding * 2
    image_height = text_height + padding * 2

    # Create the final image with the calculated size
    image = Image.new('RGB', (image_width, image_height), color=(0, 0, 0))
    draw = ImageDraw.Draw(image)
    # Calculate the position of the text to center it
    position = (padding, padding)

    # Add text to the image
    draw.text(position, bidi_text, (255, 255, 255), font=font)
    image = image.resize(image_dimensions_px)
    image.save(output_image_path)


def save_text_to_file(text, filepath):
    """
    Saves a given string to a text file.

    Args:
        text (str): The text to save.
        filepath (str): The path to the output file.
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
    except Exception as e:
        print(f"Error saving text to {filepath}: {e}")


if __name__ == "__main__":
    # --- Configuration ---
    num_samples_to_generate = 10000
    base_output_directory = "DataSet/NewSynthetic"  # Main output directory
    EMPTY_AUGMENT_PROBABILITY = 0.2 # Probability that an augmenting phrase will be empty

    # Define subdirectories for images, matrices, and text lines
    output_images_dir = os.path.join(base_output_directory, "images")
    output_matrices_dir = os.path.join(base_output_directory, "matrices")
    output_similarity_matrices_dir = os.path.join(base_output_directory, "similarity_matrices")
    output_diff_nw_matrices_dir = os.path.join(base_output_directory, "diff_nw_matrices")
    output_text_lines_dir = os.path.join(base_output_directory, "texts")

    # Create output directories if they don't exist
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

    # Initialize DiffNWAlgo model
    diff_nw_algo = DiffNWAlgo(match_score=matchScore, miss_score=mismatchScore, gap=gapScore, batch=False)

    # --- Define Arabic phrases for sentence construction ---
    base_arabic_phrases = [
        "الشمس مشرقة اليوم حقاً",  # The sun is really shining today
        "القهوة صباحاً تعدل المزاج",  # Morning coffee adjusts the mood
        "الحديقة مليئة بالزهور الملونة",  # The park is full of colorful flowers
        "القراءة غذاء العقل والروح",  # Reading is food for the mind and soul
        "السفر يفتح آفاقاً جديدة",  # Travel opens new horizons
        "الرياضة تحافظ على صحة الجسم",  # Sports maintain body health
        "الموسيقى الهادئة تريح الأعصاب",  # Calm music soothes the nerves
        "العمل الجاد يؤتي ثماره دائماً",  # Hard work always pays off
        "التكنولوجيا تسهل حياتنا كثيراً",  # Technology greatly facilitates our lives
        "التعاون يحقق أفضل النتائج الممكنة",  # Cooperation achieves the best possible results
        "الأزهار الجميلة تتفتح في الربيع",  # Beautiful flowers bloom in spring
        "البحر يبدو اليوم جميلاً وهادئاً",  # The sea looks beautiful and calm today
        "التاريخ يعلمنا دروساً قيمة ومهمة",  # History teaches us valuable and important lessons
        "البيئة الطبيعية تحتاج إلى رعايتنا جميعاً",  # The natural environment needs all our care
        "الأمل الصادق يعطينا القوة للاستمرار",  # Sincere hope gives us strength to continue
        "القصص المسلية ممتعة جداً للأطفال",  # Entertaining stories are very fun for children
        "أوراق الشجر الخضراء تتساقط خريفاً",  # Green tree leaves fall in autumn
        "العائلة هي السند الحقيقي وقت الشدة",  # Family is the true support in times of hardship
        "الابتسامة الصادقة تنشر السعادة حولنا",  # A sincere smile spreads happiness around us
        "النجوم اللامعة تلمع في السماء ليلاً",  # Bright stars shine in the sky at night
        "الأحلام الكبيرة تحتاج إلى سعي وجهد",  # Big dreams need pursuit and effort
        "الفصول الأربعة تتغير بانتظام دائماً",  # The four seasons always change regularly
        "الطعام العربي التقليدي لذيذ جداً",  # Traditional Arabic food is very delicious
        "الاستماع الجيد يعتبر مهارة مهمة",  # Good listening is considered an important skill
        "الصداقة الحقيقية كنز ثمين بالفعل",  # True friendship is indeed a precious treasure
        "الجو لطيف ومشمس هذا الصباح",  # The weather is nice and sunny this morning
        "الكتاب الجديد يبدو مثيراً للاهتمام",  # The new book looks interesting
        "الهدوء يعم المكان الآن",  # Calmness pervades the place now
        "الأخبار السارة أسعدت الجميع",  # The good news made everyone happy
        "الوقت يمر بسرعة كبيرة جداً",  # Time passes very quickly
    ]

    augmenting_arabic_phrases = [
        "بكل تأكيد",  # Certainly
        "في بعض الأحيان",  # Sometimes
        "بشكل عام",  # In general
        "على الأغلب الظن",  # Most likely
        "في الواقع تماماً",  # Actually / In reality, exactly
        "لهذا السبب بالذات",  # For this very reason
        "مع الأسف الشديد",  # Unfortunately / With deep regret
        "لحسن الحظ فعلاً",  # Fortunately, indeed
        "بعد قليل من الوقت",  # After a little while
        "منذ فترة ليست بالطويلة",  # Since not too long ago
        "حتى الآن فقط",  # Only until now / So far only
        "في المساء الباكر",  # In the early evening
        "قبل الظهر مباشرة",  # Right before noon
        "أثناء النهار كله",  # During the whole day
        "ببطء شديد وحذر",  # Very slowly and carefully
        "بسرعة كبيرة جداً",  # Very quickly indeed
        "بدون أدنى شك",  # Without the slightest doubt
        "على الإطلاق أبداً",  # Absolutely never / Not at all
        "مرة أخرى قريباً",  # Once again soon
        "في هذا المكان الجميل",  # In this beautiful place
        "هناك أيضاً سبب آخر",  # There is also another reason
        "على سبيل المثال لا الحصر",  # For example, but not limited to
        "فوق كل شيء آخر",  # Above everything else
        "تحت أي ظرف كان",  # Under any circumstance whatsoever
        "بجوار النافذة الكبيرة",  # Next to the large window
        "في الحقيقة",  # In truth / In fact
        "غالباً ما",  # Often
        "ربما لاحقاً",  # Maybe later
        "بالتأكيد نعم",  # Definitely yes
        "خصوصاً اليوم",  # Especially today
    ]

    # Font settings
    font_to_use = "Fonts/Amiri-Regular.ttf"  # Ensure this path is correct
    text_font_size = 90

    # Image settings
    img_width = 1024  # This is used for image_dimensions, but create_arabic_text_image recalculates
    img_height = 128
    image_dimensions = (img_width, img_height)  # Passed but effectively overridden by text size + padding

    # Smith-Waterman parameters
    NW_match_score = matchScore
    NW_mismatch_penalty = mismatchScore
    NW_gap_penalty = gapScore

    print(f"\nStarting generation of {num_samples_to_generate} sample pairs (image + matrix + text lines)...")

    def get_augment_phrase(phrases_list, empty_prob):
        if not phrases_list:  # Handle empty augmenting list
            return ""
        if random.random() < empty_prob:
            return ""
        return random.choice(phrases_list)

    for i in tqdm(range(1, num_samples_to_generate + 1), desc="Generating Samples"):
        # --- Sentence generation: aligned phrase at START of sentence 1,
        #     END of sentence 2.
        #
        # Arabic is RTL, so:
        #   "start of sentence"  = rightmost region of the image  = patch 0
        #   "end of sentence"    = leftmost  region of the image  = patch S-1
        #
        # Layout:
        #   sentence1:  [ common_phrase ]  [ aug_tail1 ]
        #   sentence2:  [ aug_head2 ]  [ common_phrase ]
        #
        # LENGTH VARIATION: each sentence gets a *random* number of
        # augmenting phrases (between 1 and 4) so the two sentences
        # end up with genuinely different lengths most of the time.
        # -----------------------------------------------------------
        common_phrase = random.choice(base_arabic_phrases)

        # --- Random number of augmenting parts per sentence (1-4) ---
        num_aug_tail1 = random.randint(1, 4)
        num_aug_head2 = random.randint(1, 4)

        # Extra context appended AFTER the common phrase in sentence 1
        aug_tail1_parts = [
            get_augment_phrase(augmenting_arabic_phrases, EMPTY_AUGMENT_PROBABILITY)
            for _ in range(num_aug_tail1)
        ]
        # Force at least one non-empty tail so there is genuinely extra content
        while not any(p.strip() for p in aug_tail1_parts):
            aug_tail1_parts = [random.choice(augmenting_arabic_phrases)] + [
                get_augment_phrase(augmenting_arabic_phrases, EMPTY_AUGMENT_PROBABILITY)
                for _ in range(num_aug_tail1 - 1)
            ]

        # Extra context prepended BEFORE the common phrase in sentence 2
        aug_head2_parts = [
            get_augment_phrase(augmenting_arabic_phrases, EMPTY_AUGMENT_PROBABILITY)
            for _ in range(num_aug_head2)
        ]
        while not any(p.strip() for p in aug_head2_parts):
            aug_head2_parts = [random.choice(augmenting_arabic_phrases)] + [
                get_augment_phrase(augmenting_arabic_phrases, EMPTY_AUGMENT_PROBABILITY)
                for _ in range(num_aug_head2 - 1)
            ]

        sentence1_parts = [common_phrase] + aug_tail1_parts
        sentence2_parts = aug_head2_parts + [common_phrase]

        # Join parts, filtering out empty strings and stripping extra spaces
        arabic_sentence_1 = " ".join(part.strip() for part in sentence1_parts if part.strip())
        arabic_sentence_2 = " ".join(part.strip() for part in sentence2_parts if part.strip())

        # Define output file paths for images
        output_img_file_1 = os.path.join(output_images_dir, f"img1_{i}.png")
        output_img_file_2 = os.path.join(output_images_dir, f"img2_{i}.png")

        # Define output file path for the matrix (now .npy)
        output_matrix_file = os.path.join(output_matrices_dir, f"scoreMatrix_{i}.npy")
        output_similarity_matrix_file = os.path.join(output_similarity_matrices_dir, f"similarityMatrix_{i}.npy")
        output_diff_nw_matrix_file = os.path.join(output_diff_nw_matrices_dir, f"diffNWMatrix_{i}.npy")

        # Define output file paths for the text lines
        output_text_file_1 = os.path.join(output_text_lines_dir, f"text1_{i}.txt")
        output_text_file_2 = os.path.join(output_text_lines_dir, f"text2_{i}.txt")

        # 1. Generate and save images
        create_arabic_text_image(arabic_sentence_1, font_to_use, text_font_size, image_dimensions, output_img_file_1)
        create_arabic_text_image(arabic_sentence_2, font_to_use, text_font_size, image_dimensions, output_img_file_2)

        # 2. Generate and save Needleman-Wunsch scoring matrix as .npy
        seq1_no_spaces = arabic_sentence_1.replace(" ", "")
        seq2_no_spaces = arabic_sentence_2.replace(" ", "")
        
        scoring_matrix = needleman_wunsch_matrix(
            seq1_no_spaces,
            seq2_no_spaces,
            match_score=NW_match_score,
            mismatch_penalty=NW_mismatch_penalty,
            gap_penalty=NW_gap_penalty
        )
        
        # 3. Generate and save similarity matrix (1 for match, 0 for mismatch)
        similarity_matrix = create_similarity_matrix(seq1_no_spaces, seq2_no_spaces)
        save_matrix_to_file(similarity_matrix, output_similarity_matrix_file)
        
        # 4. Generate and save DiffNW alignment matrix
        # Convert similarity matrix to torch tensor with proper scaling for DiffNWAlgo
        sim_tensor = torch.tensor(similarity_matrix, dtype=torch.float32)
        # Scale: 1 (match) -> matchScore, 0 (mismatch) -> mismatchScore
        scaled_sim = sim_tensor * (matchScore - mismatchScore) + mismatchScore
        # Add batch dimension: [1, N, M]
        # scaled_sim_batch = scaled_sim.unsqueeze(0)
        scaled_sim_batch = scaled_sim
        
        with torch.no_grad():
            diff_nw_align = diff_nw_algo(similarity_matrix=scaled_sim_batch, calc_cosine=False)
        
        # Save the diff NW alignment matrix
        diff_nw_matrix = diff_nw_align.squeeze(0).cpu().numpy()
        save_matrix_to_file(diff_nw_matrix, output_diff_nw_matrix_file)
        
        fig, axes = plt.subplots(1, 1, figsize=(20, 10))
        
        sns.heatmap(scoring_matrix, cmap='jet', linewidths=0.1, linecolor='black',
                    ax=axes, yticklabels=list(seq1_no_spaces),
                    xticklabels=list(seq2_no_spaces))
        
        # plt.show()
        save_matrix_to_file(scoring_matrix, output_matrix_file)
        plt.close(fig)  # Close the figure to prevent resource leaks

        # 4. Save the raw Arabic text lines to .txt files
        save_text_to_file(arabic_sentence_1, output_text_file_1)
        save_text_to_file(arabic_sentence_2, output_text_file_2)

    print(
        f"\nScript finished. {num_samples_to_generate * 2} images, {num_samples_to_generate} matrices (as .npy), and {num_samples_to_generate * 2} text lines generated in '{base_output_directory}'.")