import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# Import sw_with_gap from DiffSWAlgo
from DiffSWAlgo import sw_with_gap


def main():
    # Create a symmetric similarity matrix (must be square). Using 4x4.
    # Binary similarity on the diagonal; symmetric off-diagonals.
    S = jnp.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
    ], dtype=jnp.float32)

    # Build the differentiable SW (non-batched) with gap penalty
    traceback = sw_with_gap(batch=False, gap_penalty=-1)

    # Run the function: returns gradient of max-score wrt S
    grad_S = traceback(S)

    # Print matrices
    # print("Similarity matrix S [4,4] (symmetric):")
    # print(np.array(S))
    # print("\nDiffSW 'traceback' output (gradient wrt S) [5,4]:")
    # print(np.array(grad_S))

    # Visualize similarity and gradient as heatmaps
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(np.array(S), cmap='coolwarm', aspect='auto')
    plt.title('Similarity Matrix S')
    plt.colorbar()
    for (i, j), val in np.ndenumerate(np.array(S)):
        plt.text(j, i, f"{val:.2f}", ha='center', va='center', color='black', fontsize=8)

    plt.subplot(1,2,2)
    plt.imshow(np.array(grad_S), cmap='viridis', aspect='auto')
    plt.title('Traceback Gradient')
    plt.colorbar()
    for (i, j), val in np.ndenumerate(np.array(grad_S)):
        plt.text(j, i, f"{val:.2f}", ha='center', va='center', color='white' if val < np.mean(grad_S) else 'black', fontsize=8)

    plt.tight_layout()
    plt.savefig('sw_demo_heatmaps.png', dpi=150)
    plt.close()


if __name__ == '__main__':
    main()
