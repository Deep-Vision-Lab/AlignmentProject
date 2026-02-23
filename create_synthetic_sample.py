import os
import random
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display

# Configurations
OUTPUT_DIR = "TestSample"
IMAGE_FILENAME = "sample_image.png"
TEXT_FILENAME = "sample_text.txt"
WINDOWS_DIR = "windows"
FONT_PATH = "Fonts/Amiri-Regular.ttf"  # From generateDataArabic.py
FONT_SIZE = 90 # From generateDataArabic.py
PADDING = 20

# Sliding Window Parameters
WINDOW_WIDTH = 22 # From Parameters.py
WINDOW_HEIGHT = 128 # The image height we target
STRIDE_RATIO = 0.5
STRIDE = int(WINDOW_WIDTH * STRIDE_RATIO)

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
windows_output_dir = os.path.join(OUTPUT_DIR, WINDOWS_DIR)
if not os.path.exists(windows_output_dir):
    os.makedirs(windows_output_dir)

# Helper function 
def create_arabic_text_image(text, font_path, font_size, output_path):
    # Reshape and bidi
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)

    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Error: Font not found at {font_path}")
        return None

    # Temporary image to calculate text size
    dummy_img = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    
    # Calculate bounding box
    left, top, right, bottom = draw.textbbox((0, 0), bidi_text, font=font)
    text_width = right - left
    text_height = bottom - top

    # Create image with padding
    img_width = text_width + PADDING * 2
    img_height = text_height + PADDING * 2
    
    # We want a fixed height for the sliding window to work predictably, usually
    # But generateDataArabic.py resizes to (1024, 128). Let's follow that pattern or keep aspect ratio?
    # The requirement asks to generate "its image".
    # If we stick to project conventions, let's target the height.
    
    target_height = WINDOW_HEIGHT
    
    # Create the image
    image = Image.new('RGB', (img_width, img_height), color=(0, 0, 0)) # Black background
    draw = ImageDraw.Draw(image)
    
    # Draw text centered
    # Position logic from generateDataArabic seems to be (padding, padding)
    # But textbbox might have negative start values.
    # To properly center, we can use the offset.
    
    draw.text((PADDING - left, PADDING - top), bidi_text, font=font, fill=(255, 255, 255)) # White text

    # Resize to target height while maintaining aspect ratio, or specific width?
    # generateDataArabic uses: image = image.resize(image_dimensions_px) where image_dimensions_px = (1024, 128)
    # This distorts the text if not careful.
    # Let's resize height to target_height and scale width proportionally.
    
    scale_factor = target_height / img_height
    new_width = int(img_width * scale_factor)
    image = image.resize((new_width, target_height), Image.Resampling.LANCZOS)
    
    image.save(output_path)
    return image

def apply_sliding_window(image, window_width, stride, output_dir):
    width, height = image.size
    count = 0
    windows = []
    
    # Slide across width
    for x in range(0, width - window_width + 1, stride):
        # Crop box is (left, top, right, bottom)
        box = (x, 0, x + window_width, height)
        window = image.crop(box)
        
        window_filename = f"window_{count:04d}.png"
        window_path = os.path.join(output_dir, window_filename)
        window.save(window_path)
        windows.append(window_path)
        count += 1
        
    print(f"Generated {count} windows in {output_dir}")
    return windows

# Main execution
if __name__ == "__main__":
    # Arabic sentence
    sentence = "الشمس مشرقة اليوم حقاً" # "The sun is really shining today"
    
    print(f"Processing sentence: {sentence}")
    
    # Save Text
    text_path = os.path.join(OUTPUT_DIR, TEXT_FILENAME)
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write(sentence)
    print(f"Saved text to {text_path}")
    
    # Generate and Save Image
    image_path = os.path.join(OUTPUT_DIR, IMAGE_FILENAME)
    image = create_arabic_text_image(sentence, FONT_PATH, FONT_SIZE, image_path)
    
    if image:
        print(f"Saved image to {image_path}")
        
        # Apply Sliding Window
        apply_sliding_window(image, WINDOW_WIDTH, WINDOW_WIDTH, windows_output_dir)
    else:
        print("Failed to generate image.")
