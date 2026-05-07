import numpy as np
import cv2

def simulate_length_discrepancy(arabic_patches, english_patches, ratio=3.0):
    """
    Simulates highly skewed extraction (Short Arabic segment vs heavily redundant English translation).
    This breaks the typical DTW 1:1 diagonal step path.
    We test if the Dropout and Multi-Scale windows handle massive repetitions.
    Returns: Two arrays mimicking patch embeddings, but with one padded/interpolated largely.
    """
    # Duplicate english patches artificially to simulate this length discrepancy
    skewe_english_len = int(len(english_patches) * ratio)
    # Simple interpolation to expand dataset
    indices = np.linspace(0, len(english_patches)-1, skewe_english_len).astype(int)
    skewed_english = english_patches[indices]
    
    return arabic_patches, skewed_english
    
def simulate_degraded_ink(image_patch, mode="noise", severity=0.3):
    """
    Adds artificial fading or noise to the test image patch to prove Dropout/Multi-Scale robustness.
    mode: "noise" or "fade"
    severity: 0.0 to 1.0 (1.0 = completely destroyed image)
    """
    img = image_patch.copy().astype(np.float32)
    
    if mode == "noise":
        noise = np.random.normal(scale=severity * 255, size=img.shape)
        img = np.clip(img + noise, 0, 255)
    elif mode == "fade":
        # Simulate fading by blending towards white (paper background)
        white = np.ones_like(img) * 255
        img = img * (1 - severity) + white * severity
        
    return img.astype(np.uint8)

if __name__ == "__main__":
    print("Executing Phase 4: Stress Testing (Finding the Limits)")
    print("Test 1: Length Discrepancy Check (Short Arabic vs Long English)")
    ar_patches = np.random.rand(50, 256)
    en_patches = np.random.rand(50, 256)
    ar, skewed_en = simulate_length_discrepancy(ar_patches, en_patches, ratio=3.0)
    print(f"Original shape: {ar_patches.shape} vs {en_patches.shape}")
    print(f"Skewed shape: {ar.shape} vs {skewed_en.shape}")
    # TODO: Pass `ar` and `skewed_en` through the D3TW distance calculator.
    
    print("\nTest 2: Degraded Ink Stress Test")
    dummy_patch = np.zeros((32, 32, 3), dtype=np.uint8)
    dummy_patch[10:20, 10:20] = 255 # White rectangle inside black patch
    
    noisy_patch = simulate_degraded_ink(dummy_patch, mode="noise", severity=0.5)
    faded_patch = simulate_degraded_ink(dummy_patch, mode="fade", severity=0.7)
    print(f"Generated noisy and faded patches. Ready for inference testing.")
    # TODO: Run inference on degraded datasets to evaluate Recall@1 drop.
