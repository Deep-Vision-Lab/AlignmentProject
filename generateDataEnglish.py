# english_image_generator.py
# Description: Generates pairs of images with English text and their corresponding
#              Smith-Waterman alignment scoring matrices.
# Dependencies: Pillow, tqdm, numpy
# To install: pip install Pillow tqdm numpy
# Usage: python generateDataEnglish.py --start 1 --end 1000 --threads 4

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
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

warnings.filterwarnings("ignore")

# Thread-local storage for random state and model instances
thread_local = threading.local()
matplotlib_lock = threading.Lock()

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
                             text_color=(255, 255, 255), background_color=(0, 0, 0), padding=20):
    try:
        font = ImageFont.truetype(font_path_or_name, font_size_px)
    except IOError:
        print(f"Critical: Fallback fonts not found for {output_image_path}. Using PIL's default font.")
        font = ImageFont.load_default()
    
    # Initial large image to draw text
    image = Image.new("RGB", (2000, 500), background_color)
    draw = ImageDraw.Draw(image)
    
    # Get text bounding box
    left, top, right, bottom = draw.textbbox((0, 0), text_to_render, font=font)
    text_width = right - left
    text_height = bottom - top
    
    # Create final image with padding
    image_width = text_width + padding * 2
    image_height = text_height + padding * 2
    image = Image.new('RGB', (image_width, image_height), color=background_color)
    draw = ImageDraw.Draw(image)
    
    position = (padding, padding)
    draw.text(position, text_to_render, text_color, font=font)
    
    # Resize to target dimensions
    image = image.resize(image_dimensions_px)
    image.save(output_image_path)

def save_text_to_file(text, filepath):
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
    except Exception as e:
        print(f"Error saving text to {filepath}: {e}")

# --- Configuration Constants ---
BASE_OUTPUT_DIRECTORY = "DataSet/Synthetic_English"
EMPTY_AUGMENT_PROBABILITY = 0.2

BASE_ENGLISH_PHRASES = [
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

AUGMENTING_ENGLISH_PHRASES = [
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

FONT_TO_USE = "Fonts/DejaVuSans.ttf"
TEXT_FONT_SIZE = 60
IMG_WIDTH = 1024
IMG_HEIGHT = 128
IMAGE_DIMENSIONS = (IMG_WIDTH, IMG_HEIGHT)
NW_MATCH_SCORE = matchScore
NW_MISMATCH_PENALTY = mismatchScore
NW_GAP_PENALTY = gapScore


def get_thread_random():
    """Get or create a thread-local random instance with unique seed."""
    if not hasattr(thread_local, 'random'):
        # Create unique seed based on thread id and current time
        seed = hash((threading.current_thread().ident, os.getpid(), random.random()))
        thread_local.random = random.Random(seed)
    return thread_local.random


def get_thread_diff_nw_algo():
    """Get or create a thread-local DiffNWAlgo instance."""
    if not hasattr(thread_local, 'diff_nw_algo'):
        # Each thread gets its own instance to avoid state sharing issues
        thread_local.diff_nw_algo = DiffNWAlgo(
            match_score=matchScore, 
            miss_score=mismatchScore, 
            gap=gapScore, 
            batch=False
        )
    return thread_local.diff_nw_algo


def get_augment_phrase(phrases_list, empty_prob, rng=None):
    """Get an augmentation phrase with probability of returning empty."""
    if rng is None:
        rng = random
    if not phrases_list:
        return ""
    if rng.random() < empty_prob:
        return ""
    return rng.choice(phrases_list)


def generate_single_sample(i, output_dirs):
    """Generate a single sample (images, matrices, text files) for index i."""
    output_images_dir, output_matrices_dir, output_similarity_matrices_dir, \
        output_diff_nw_matrices_dir, output_text_lines_dir = output_dirs
    
    # Use thread-local random for unique sentences per thread
    rng = get_thread_random()
    # Use thread-local model instance
    diff_nw_algo = get_thread_diff_nw_algo()
    
    # Select common phrases
    if len(BASE_ENGLISH_PHRASES) >= 2:
        common_phrases = rng.sample(BASE_ENGLISH_PHRASES, 2)
        common_phrase1 = common_phrases[0]
        common_phrase2 = common_phrases[1]
    elif len(BASE_ENGLISH_PHRASES) == 1:
        common_phrase1 = BASE_ENGLISH_PHRASES[0]
        common_phrase2 = BASE_ENGLISH_PHRASES[0]
    else:
        common_phrase1 = "Base phrase one"
        common_phrase2 = "Base phrase two"
    
    # Get augmentation phrases
    aug1_prefix = get_augment_phrase(AUGMENTING_ENGLISH_PHRASES, EMPTY_AUGMENT_PROBABILITY, rng)
    aug1_middle = get_augment_phrase(AUGMENTING_ENGLISH_PHRASES, EMPTY_AUGMENT_PROBABILITY, rng)
    aug1_suffix = get_augment_phrase(AUGMENTING_ENGLISH_PHRASES, EMPTY_AUGMENT_PROBABILITY, rng)
    aug2_prefix = get_augment_phrase(AUGMENTING_ENGLISH_PHRASES, EMPTY_AUGMENT_PROBABILITY, rng)
    aug2_middle = get_augment_phrase(AUGMENTING_ENGLISH_PHRASES, EMPTY_AUGMENT_PROBABILITY, rng)
    aug2_suffix = get_augment_phrase(AUGMENTING_ENGLISH_PHRASES, EMPTY_AUGMENT_PROBABILITY, rng)
    
    # Build sentences
    sentence1_parts = [aug1_prefix, common_phrase1, aug1_middle, common_phrase2, aug1_suffix]
    sentence2_parts = [aug2_prefix, common_phrase1, aug2_middle, common_phrase2, aug2_suffix]
    english_sentence_1 = " ".join(part.strip() for part in sentence1_parts if part.strip())
    english_sentence_2 = " ".join(part.strip() for part in sentence2_parts if part.strip())
    
    if not english_sentence_1:
        english_sentence_1 = common_phrase1
    if not english_sentence_2:
        english_sentence_2 = common_phrase2
    
    # Random swap
    if rng.choice([True, False]):
        english_sentence_1, english_sentence_2 = english_sentence_2, english_sentence_1
    
    # Define output file paths
    output_img_file_1 = os.path.join(output_images_dir, f"img1_{i}.png")
    output_img_file_2 = os.path.join(output_images_dir, f"img2_{i}.png")
    output_matrix_file = os.path.join(output_matrices_dir, f"scoreMatrix_{i}.npy")
    output_similarity_matrix_file = os.path.join(output_similarity_matrices_dir, f"similarityMatrix_{i}.npy")
    output_diff_nw_matrix_file = os.path.join(output_diff_nw_matrices_dir, f"diffNWMatrix_{i}.npy")
    output_text_file_1 = os.path.join(output_text_lines_dir, f"text1_{i}.txt")
    output_text_file_2 = os.path.join(output_text_lines_dir, f"text2_{i}.txt")
    
    # Create images
    create_english_text_image(english_sentence_1, FONT_TO_USE, TEXT_FONT_SIZE, IMAGE_DIMENSIONS, output_img_file_1)
    create_english_text_image(english_sentence_2, FONT_TO_USE, TEXT_FONT_SIZE, IMAGE_DIMENSIONS, output_img_file_2)
    
    # Compute matrices
    seq1_no_spaces = english_sentence_1.replace(" ", "")
    seq2_no_spaces = english_sentence_2.replace(" ", "")
    
    scoring_matrix = needleman_wunsch_matrix(
        seq1_no_spaces,
        seq2_no_spaces,
        match_score=NW_MATCH_SCORE,
        mismatch_penalty=NW_MISMATCH_PENALTY,
        gap_penalty=NW_GAP_PENALTY
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
    
    # Save heatmap and matrix - Matplotlib is not thread-safe, use a lock
    with matplotlib_lock:
        fig, axes = plt.subplots(1, 1, figsize=(20, 10))
        sns.heatmap(scoring_matrix, cmap='jet', linewidths=0.1, linecolor='black',
                    ax=axes, yticklabels=list(seq1_no_spaces),
                    xticklabels=list(seq2_no_spaces))
        save_matrix_to_file(scoring_matrix, output_matrix_file)
        plt.close(fig)
    
    # Save text files
    save_text_to_file(english_sentence_1, output_text_file_1)
    save_text_to_file(english_sentence_2, output_text_file_2)
    
    return i


def generate_range_worker(start_idx, end_idx, output_dirs, progress_callback=None):
    """Worker function to generate samples for a specific range [start_idx, end_idx]."""
    results = []
    for i in range(start_idx, end_idx + 1):
        try:
            generate_single_sample(i, output_dirs)
            results.append((i, True, None))
            if progress_callback:
                progress_callback(1)
        except Exception as e:
            results.append((i, False, str(e)))
    return results


def generate_data_multithreaded(start_idx, end_idx, num_threads=4, base_output_directory=None):
    """
    Generate data using multiple threads.
    
    Args:
        start_idx: Starting index (inclusive)
        end_idx: Ending index (inclusive)
        num_threads: Number of threads to use
        base_output_directory: Output directory path
    """
    if base_output_directory is None:
        base_output_directory = BASE_OUTPUT_DIRECTORY
    
    # Setup directories
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
    
    output_dirs = (output_images_dir, output_matrices_dir, output_similarity_matrices_dir,
                   output_diff_nw_matrices_dir, output_text_lines_dir)
    
    print(f"Output will be saved in: {os.path.abspath(base_output_directory)}")
    print(f"Generating samples from index {start_idx} to {end_idx}")
    print(f"Using {num_threads} threads")
    
    total_samples = end_idx - start_idx + 1
    
    # Calculate ranges for each thread
    samples_per_thread = total_samples // num_threads
    ranges = []
    current_start = start_idx
    
    for t in range(num_threads):
        if t == num_threads - 1:
            # Last thread takes the remainder
            current_end = end_idx
        else:
            current_end = current_start + samples_per_thread - 1
        ranges.append((current_start, current_end))
        current_start = current_end + 1
    
    print(f"\nThread ranges:")
    for t, (s, e) in enumerate(ranges):
        print(f"  Thread {t + 1}: indices {s} to {e} ({e - s + 1} samples)")
    
    # Progress bar with thread-safe update
    pbar = tqdm(total=total_samples, desc="Generating Samples")
    pbar_lock = threading.Lock()
    
    def update_progress(n=1):
        with pbar_lock:
            pbar.update(n)
    
    # Execute with thread pool
    all_results = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for start, end in ranges:
            future = executor.submit(
                generate_range_worker, 
                start, end, output_dirs, update_progress
            )
            futures.append(future)
        
        for future in as_completed(futures):
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                print(f"Thread error: {e}")
    
    pbar.close()
    
    # Report results
    successful = sum(1 for _, success, _ in all_results if success)
    failed = sum(1 for _, success, _ in all_results if not success)
    
    print(f"\nGeneration complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    
    if failed > 0:
        print("\nFailed samples:")
        for idx, success, error in all_results:
            if not success:
                print(f"  Sample {idx}: {error}")
    
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate English text image pairs with alignment matrices using multi-threading.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 3000 samples using 4 threads (default)
  python generateDataEnglish.py
  
  # Generate samples from index 1 to 1000 using 8 threads
  python generateDataEnglish.py --start 1 --end 1000 --threads 8
  
  # Generate samples from index 5001 to 10000 (useful for parallel runs)
  python generateDataEnglish.py --start 5001 --end 10000 --threads 4
  
  # Custom output directory
  python generateDataEnglish.py --start 1 --end 500 --output DataSet/Custom_English
        """
    )
    
    parser.add_argument(
        "--start", "-s",
        type=int,
        default=1,
        help="Starting index for sample generation (default: 1)"
    )
    parser.add_argument(
        "--end", "-e",
        type=int,
        default=3000,
        help="Ending index for sample generation (default: 3000)"
    )
    parser.add_argument(
        "--threads", "-t",
        type=int,
        default=4,
        help="Number of threads to use (default: 4)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=BASE_OUTPUT_DIRECTORY,
        help=f"Base output directory (default: {BASE_OUTPUT_DIRECTORY})"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.start < 1:
        parser.error("Start index must be >= 1")
    if args.end < args.start:
        parser.error("End index must be >= start index")
    if args.threads < 1:
        parser.error("Number of threads must be >= 1")
    
    num_samples = args.end - args.start + 1
    print(f"\n{'='*60}")
    print(f"English Data Generator - Multi-threaded")
    print(f"{'='*60}")
    print(f"Range: {args.start} to {args.end} ({num_samples} samples)")
    print(f"Threads: {args.threads}")
    print(f"Output: {args.output}")
    print(f"{'='*60}\n")
    
    # Run multi-threaded generation
    generate_data_multithreaded(
        start_idx=args.start,
        end_idx=args.end,
        num_threads=args.threads,
        base_output_directory=args.output
    )
    
    print(f"\nScript finished. {num_samples * 2} images, {num_samples} matrices (as .npy), "
          f"and {num_samples * 2} text lines generated in '{args.output}'.")
