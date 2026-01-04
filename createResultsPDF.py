import os
from fpdf import FPDF
from PIL import Image
import re
import logging
import argparse
import warnings_filter  # noqa: F401
    

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TrainResultsPDF(FPDF):
    def header(self):
        pass
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def get_epochs(base_path):
    """Get sorted list of epoch folders"""
    epochs = []
    if os.path.exists(base_path):
        for item in os.listdir(base_path):
            if item.startswith('Epoch_'):
                epochs.append(item)
    return sorted(epochs, key=lambda x: int(x.split('_')[1]))

def get_samples(epoch_path):
    """Get sorted list of sample folders (0, 1, etc.)"""
    samples = []
    if os.path.exists(epoch_path):
        for item in os.listdir(epoch_path):
            if item.isdigit():
                samples.append(item)
    return sorted(samples, key=int)

def get_one_sample(epoch_path):
    """Get just one sample folder (lowest number)"""
    samples = []
    if os.path.exists(epoch_path):
        for item in os.listdir(epoch_path):
            if item.isdigit():
                samples.append(item)
    if samples:
        return [sorted(samples, key=int)[0]]
    return []

def add_title_page(pdf):
    """Add title page"""
    pdf.add_page()
    pdf.set_font('Arial', 'B', 36)
    pdf.set_y(120)
    pdf.cell(0, 20, 'Train Results', 0, 1, 'C')
    pdf.set_font('Arial', '', 14)
    pdf.cell(0, 10, 'Alignment Project Visualizations', 0, 1, 'C')

def add_section_header(pdf, section_title):
    """Add section header on new page"""
    pdf.add_page()
    pdf.set_font('Arial', 'B', 20)
    pdf.set_y(20)
    pdf.cell(0, 15, section_title, 0, 1, 'L')
    pdf.ln(5)

def add_subsection(pdf, subsection_title):
    """Add subsection header"""
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, subsection_title, 0, 1, 'L')
    pdf.ln(2)

def add_image_scaled(pdf, image_path, x, y, max_w, max_h, center=True):
    """Add image with proper scaling to fit within max dimensions.
    By default, centers the image horizontally on the page."""
    if not os.path.exists(image_path):
        return y
    try:
        img = Image.open(image_path)
        width, height = img.size
        ratio = min(max_w / width, max_h / height)
        new_w = width * ratio
        new_h = height * ratio
        
        if center:
            x = (pdf.w - new_w) / 2
            
        pdf.image(image_path, x=x, y=y, w=new_w, h=new_h)
        return y + new_h
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return y

def add_title_and_global_images(pdf, input_images_path):
    """Add title page, then global Line 1, Line 1 Windows, Line 2, Line 2 Windows."""
    # Title page
    pdf.add_page()
    pdf.set_font('Arial', 'B', 36)
    pdf.set_y(120)
    pdf.cell(0, 20, 'Train Results', 0, 1, 'C')
    pdf.set_font('Arial', '', 14)
    pdf.cell(0, 10, 'Alignment Project Visualizations', 0, 1, 'C')

    # Use first epoch and first sample for global images
    epochs = get_epochs(input_images_path)
    if not epochs:
        return
    epoch_path = os.path.join(input_images_path, epochs[0])
    samples = get_one_sample(epoch_path)
    if not samples:
        return
    sample_path = os.path.join(epoch_path, samples[0])

    # Global Input Images page
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Global Input Images', 0, 1, 'L')
    pdf.ln(2)
    # Line 1
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, 'Line 1:', 0, 1, 'C')
    img1_path = os.path.join(sample_path, 'Image1.png')
    if os.path.exists(img1_path):
        add_image_scaled(pdf, img1_path, 10, pdf.get_y(), 190, 45, center=True)
        pdf.set_y(pdf.get_y() + 48)
    # Line 1 Windows
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(0, 6, 'Line 1 Windows:', 0, 1, 'L')
    img1_win_path = os.path.join(sample_path, 'Image1_SlidingWindows.png')
    if os.path.exists(img1_win_path):
        new_y = add_image_scaled(pdf, img1_win_path, 10, pdf.get_y(), pdf.w - 20, 150)
        pdf.set_y(new_y + 5)
        pdf.add_page()
    # Line 2
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, 'Line 2:', 0, 1, 'C')
    img2_path = os.path.join(sample_path, 'Image2.png')
    if os.path.exists(img2_path):
        add_image_scaled(pdf, img2_path, 10, pdf.get_y(), 190, 45, center=True)
        pdf.set_y(pdf.get_y() + 48)
    # Line 2 Windows
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(0, 6, 'Line 2 Windows:', 0, 1, 'L')
    img2_win_path = os.path.join(sample_path, 'Image2_SlidingWindows.png')
    if os.path.exists(img2_win_path):
        new_y = add_image_scaled(pdf, img2_win_path, 10, pdf.get_y(), pdf.w - 20, 150)
        pdf.set_y(new_y + 5)

def add_all_score_matrices(pdf, score_matrices_path, epochs):
    """Add all Score Matrices for all epochs, each epoch on its own page."""
    header_added = False
    for epoch in epochs:
        epoch_num = epoch.split('_')[1]  # Extract number from Epoch_X
        epoch_score_path = os.path.join(score_matrices_path, epoch)
        samples = get_one_sample(epoch_score_path)
        for sample in samples:
            sample_score_path = os.path.join(epoch_score_path, sample)
            score_matrix = os.path.join(sample_score_path, 'score_matrix', 'ScoreMatrix.png')
            score_matrix_dist = os.path.join(sample_score_path, 'score_matrix', 'ScoreMatrixDistance.png')
            if not (os.path.exists(score_matrix) or os.path.exists(score_matrix_dist)):
                continue
            # Section header once
            if not header_added:
                add_section_header(pdf, '1. Score Matrices')
                header_added = True
            else:
                pdf.add_page()
            # Epoch header
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, f'Epoch {epoch_num}', 0, 1, 'L')
            pdf.ln(2)
            # Score Matrix (own page)
            if os.path.exists(score_matrix):
                new_y = add_image_scaled(pdf, score_matrix, 10, pdf.get_y(), pdf.w - 20, 120)
                pdf.set_y(new_y + 3)
            # Distance Score Matrix (new page)
            if os.path.exists(score_matrix_dist):
                pdf.add_page()
                pdf.set_font('Arial', 'B', 14)
                pdf.cell(0, 10, f'Epoch {epoch_num} - Score Matrix Distance', 0, 1, 'L')
                pdf.ln(2)
                new_y = add_image_scaled(pdf, score_matrix_dist, 10, pdf.get_y(), pdf.w - 20, 120)
                pdf.set_y(new_y + 2)
                # Read and display distance value
                dist_txt = os.path.join(sample_score_path, 'score_matrix', 'ScoreMatrixDistanceSum.txt')
                d = None
                if os.path.exists(dist_txt):
                    try:
                        with open(dist_txt, 'r') as f:
                            d = f.read().strip()
                    except Exception:
                        d = None
                pdf.set_font('Arial', '', 12)
                if d is not None:
                    pdf.cell(0, 10, f'The Distance is: {d}', 0, 1, 'C')
                pdf.set_y(pdf.get_y() + 3)

def add_all_path_matrices(pdf, score_matrices_path, epochs):
    """Add all Path Matrices for all epochs, each epoch on its own page."""
    header_added = False
    for epoch in epochs:
        epoch_num = epoch.split('_')[1]  # Extract number from Epoch_X
        epoch_path = os.path.join(score_matrices_path, epoch)
        samples = get_one_sample(epoch_path)
        for sample in samples:
            sample_path = os.path.join(epoch_path, sample, 'path')
            path_img = os.path.join(sample_path, 'Paths.png')
            path_dist_img = os.path.join(sample_path, 'PathDistance.png')
            if not (os.path.exists(path_img) or os.path.exists(path_dist_img)):
                continue
            # Section header once
            if not header_added:
                add_section_header(pdf, '2. Path Matrices')
                header_added = True
            else:
                pdf.add_page()
            # Epoch header
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, f'Epoch {epoch_num}', 0, 1, 'L')
            pdf.ln(2)
            # Path Matrix (own page)
            if os.path.exists(path_img):
                new_y = add_image_scaled(pdf, path_img, 10, pdf.get_y(), pdf.w - 20, 120)
                pdf.set_y(new_y + 3)
            # Distance Path Matrix (new page)
            if os.path.exists(path_dist_img):
                pdf.add_page()
                pdf.set_font('Arial', 'B', 14)
                pdf.cell(0, 10, f'Epoch {epoch_num} - Path Matrix Distance', 0, 1, 'L')
                pdf.ln(2)
                new_y = add_image_scaled(pdf, path_dist_img, 10, pdf.get_y(), pdf.w - 20, 120)
                pdf.set_y(new_y + 2)
                # Read and display distance value
                dist_txt = os.path.join(sample_path, 'PathDistanceSum.txt')
                d = None
                if os.path.exists(dist_txt):
                    try:
                        with open(dist_txt, 'r') as f:
                            d = f.read().strip()
                    except Exception:
                        d = None
                pdf.set_font('Arial', '', 12)
                if d is not None:
                    pdf.cell(0, 10, f'The Distance is: {d}', 0, 1, 'C')
                pdf.set_y(pdf.get_y() + 3)

def add_all_height_vectors(pdf, score_matrices_path, epochs):
    """Add all HeightDiff Vectors for all epochs, two epochs per page, vectors and distance on same page."""
    header_added = False
    epochs_on_page = 0  # Track how many epochs on the current page
    for epoch in epochs:
        epoch_num = epoch.split('_')[1]  # Extract number from Epoch_X
        epoch_path = os.path.join(score_matrices_path, epoch)
        samples = get_one_sample(epoch_path)
        for sample in samples:
            sample_path = os.path.join(epoch_path, sample, 'HeightVectors')
            vectors_img = os.path.join(sample_path, 'VerticalVectors.png')
            vectors_dist_img = os.path.join(sample_path, 'VerticalVectorsDistance.png')
            if not (os.path.exists(vectors_img) or os.path.exists(vectors_dist_img)):
                continue
            # Section header once (starts a new page)
            if not header_added:
                add_section_header(pdf, '3. HeightDiff Vectors')
                header_added = True
                epochs_on_page = 0
            else:
                # Start new page every 2 epochs
                if epochs_on_page >= 2:
                    pdf.add_page()
                    epochs_on_page = 0
            # Epoch header
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, f'Epoch {epoch_num}', 0, 1, 'L')
            pdf.ln(2)
            # Vertical Vectors (same page)
            if os.path.exists(vectors_img):
                new_y = add_image_scaled(pdf, vectors_img, 10, pdf.get_y(), pdf.w - 20, 50)
                pdf.set_y(new_y + 2)
            # Distance Vertical Vectors (same page, right below)
            if os.path.exists(vectors_dist_img):
                new_y = add_image_scaled(pdf, vectors_dist_img, 10, pdf.get_y(), pdf.w - 20, 50)
                pdf.set_y(new_y + 2)
                # Read and display distance value
                dist_txt = os.path.join(sample_path, 'VerticalVectorsDistanceSum.txt')
                d = None
                if os.path.exists(dist_txt):
                    try:
                        with open(dist_txt, 'r') as f:
                            d = f.read().strip()
                    except Exception:
                        d = None
                pdf.set_font('Arial', '', 12)
                if d is not None:
                    pdf.cell(0, 10, f'The Distance is: {d}', 0, 1, 'C')
                pdf.set_y(pdf.get_y() + 3)
            epochs_on_page += 1
            

def add_score_matrices_section(pdf, score_path, similarity_path, max_epochs=None):
    """Add Score Matrices section with Epoch headers and score matrices.
    Lazily add the section header only if content exists to avoid empty pages."""
    epochs = get_epochs(score_path)
    if isinstance(max_epochs, int) and max_epochs > 0:
        epochs = epochs[:max_epochs]
    header_added = False

    for i, epoch in enumerate(epochs):
        epoch_score_path = os.path.join(score_path, epoch)
        samples = get_one_sample(epoch_score_path)

        for sample in samples:
            sample_score_path = os.path.join(epoch_score_path, sample)

            score_matrix = os.path.join(sample_score_path, 'score_matrix', 'ScoreMatrix.png')
            score_matrix_dist = os.path.join(sample_score_path, 'score_matrix', 'ScoreMatrixDistance.png')

            # Skip if no score matrices found for this epoch/sample
            if not (os.path.exists(score_matrix) or os.path.exists(score_matrix_dist)):
                continue

            # Add section header once when first content is found
            if not header_added:
                add_section_header(pdf, '1. Score Matrices')
                header_added = True

            # Epoch header
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, f'Epoch {i}', 0, 1, 'L')
            pdf.ln(2)

            # Score Matrix
            if os.path.exists(score_matrix):
                new_y = add_image_scaled(pdf, score_matrix, 10, pdf.get_y(), pdf.w - 20, 120)
                pdf.set_y(new_y + 3)

            # Distance Score Matrix (immediately under)
            if os.path.exists(score_matrix_dist):
                new_y = add_image_scaled(pdf, score_matrix_dist, 10, pdf.get_y(), pdf.w - 20, 120)
                pdf.set_y(new_y + 5)

def add_path_matrices_section(pdf, score_path, max_epochs=None):
    """Add Path Matrices section mirroring Score Matrices layout.
    Lazily add the section header only if content exists to avoid empty pages."""
    epochs = get_epochs(score_path)
    if isinstance(max_epochs, int) and max_epochs > 0:
        epochs = epochs[:max_epochs]
    header_added = False

    for i, epoch in enumerate(epochs):
        epoch_path = os.path.join(score_path, epoch)
        samples = get_one_sample(epoch_path)

        for sample in samples:
            sample_path = os.path.join(epoch_path, sample, 'path')
            if not os.path.exists(sample_path):
                continue

            path_img = os.path.join(sample_path, 'Paths.png')
            path_dist_img = os.path.join(sample_path, 'PathDistance.png')

            # Skip if no path images found
            if not (os.path.exists(path_img) or os.path.exists(path_dist_img)):
                continue

            # Add section header once when first content is found
            if not header_added:
                add_section_header(pdf, '2. Path Matrices')
                header_added = True

            # Epoch header
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, f'Epoch {i}', 0, 1, 'L')
            pdf.ln(2)

            # Path Matrix (centered, near page width)
            if os.path.exists(path_img):
                new_y = add_image_scaled(pdf, path_img, 10, pdf.get_y(), pdf.w - 20, 120)
                pdf.set_y(new_y + 3)

            # Distance Path Matrix (immediately under, centered)
            if os.path.exists(path_dist_img):
                new_y = add_image_scaled(pdf, path_dist_img, 10, pdf.get_y(), pdf.w - 20, 120)
                pdf.set_y(new_y + 5)

def add_height_vectors_section(pdf, score_path, max_epochs=None):
    """Add HeightDiff Vectors section mirroring Score/Path layout.
    Lazily add the section header only if content exists to avoid empty pages."""
    epochs = get_epochs(score_path)
    if isinstance(max_epochs, int) and max_epochs > 0:
        epochs = epochs[:max_epochs]
    header_added = False

    for i, epoch in enumerate(epochs):
        epoch_path = os.path.join(score_path, epoch)
        samples = get_one_sample(epoch_path)

        for sample in samples:
            sample_path = os.path.join(epoch_path, sample, 'HeightVectors')

            if not os.path.exists(sample_path):
                continue

            vectors_img = os.path.join(sample_path, 'VerticalVectors.png')
            vectors_dist_img = os.path.join(sample_path, 'VerticalVectorsDistance.png')

            # Skip if no vectors images found
            if not (os.path.exists(vectors_img) or os.path.exists(vectors_dist_img)):
                continue

            # Add section header once when first content is found
            if not header_added:
                add_section_header(pdf, '3. HeightDiff Vectors')
                header_added = True

            # Epoch header
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, f'Epoch {i}', 0, 1, 'L')
            pdf.ln(2)

            # Vertical Vectors (centered, near page width)
            if os.path.exists(vectors_img):
                new_y = add_image_scaled(pdf, vectors_img, 10, pdf.get_y(), pdf.w - 20, 130)
                pdf.set_y(new_y + 3)

            # Distance Vertical Vectors (immediately under, centered)
            if os.path.exists(vectors_dist_img):
                new_y = add_image_scaled(pdf, vectors_dist_img, 10, pdf.get_y(), pdf.w - 20, 130)
                pdf.set_y(new_y + 5)

def main():
    results_dir = 'TrainResults/HeightDiff'
    output_pdf = 'Train_Results.pdf'
    parser = argparse.ArgumentParser(description='Generate Train Results PDF')
    parser.add_argument('--max-epochs', type=int, default=None, help='Limit number of epochs visualized')
    args = parser.parse_args()

    input_images_path = os.path.join(results_dir, 'InputImages', 'CNN')
    score_matrices_path = os.path.join(results_dir, 'ScoreMatricesPerEpoch', 'CNN')

    pdf = TrainResultsPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # 1. Title page and global input images (Line 1, Line 1 Windows, Line 2, Line 2 Windows)
    add_title_and_global_images(pdf, input_images_path)
    print("Phase: Title and Global Input Images done.")

    # Get all epochs (limit if requested)
    epochs = get_epochs(score_matrices_path)
    if isinstance(args.max_epochs, int) and args.max_epochs > 0:
        epochs = epochs[:args.max_epochs]

    # 2. All Score Matrices (all epochs, one per page)
    add_all_score_matrices(pdf, score_matrices_path, epochs)
    print("Phase: Score Matrices section done.")

    # 3. All Path Matrices (all epochs, one per page)
    add_all_path_matrices(pdf, score_matrices_path, epochs)
    print("Phase: Path Matrices section done.")

    # 4. All HeightDiff Vectors (all epochs, one per page)
    add_all_height_vectors(pdf, score_matrices_path, epochs)
    print("Phase: HeightDiff Vectors section done.")

    pdf.output(output_pdf)
    print(f'PDF created: {output_pdf}')

if __name__ == '__main__':
    main()
