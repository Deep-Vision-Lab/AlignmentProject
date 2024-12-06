import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import matplotlib.pylab as plt
import random
from fetch_arabic_sentence import fetch_arabic_sentence
from splitTextLine import tokenize_based_on_non_connecting_letters
from augmentSenetece import augment_sentence


def smith_waterman(seq1, seq2, match_score=2, gap_cost=-1, mismatch_cost=-6):
    """
    Perform local alignment between two sequences using the Smith-Waterman algorithm.

    :param seq1: First sequence (list of subwords in Arabic).
    :param seq2: Second sequence (list of subwords in Arabic).
    :param match_score: Score for matching characters.
    :param gap_cost: Cost (negative) for introducing a gap.
    :param mismatch_cost: Cost (negative) for mismatching characters.
    :return: The optimal local alignment score, and the alignment.
    """
    # Initialize the scoring matrix
    n = len(seq1)
    m = len(seq2)
    score_matrix = np.zeros((n + 1, m + 1))
    traceback_matrix = np.zeros((n + 1, m + 1), dtype=int)

    # Fill the scoring and traceback matrices
    max_score = 0
    max_pos = (0, 0)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                score = score_matrix[i - 1][j - 1] + match_score
            else:
                score = score_matrix[i - 1][j - 1] + mismatch_cost

            score_matrix[i][j] = max(0, score,
                                     score_matrix[i - 1][j] + gap_cost,
                                     score_matrix[i][j - 1] + gap_cost)

            if score_matrix[i][j] == 0:
                traceback_matrix[i][j] = 0
            elif score_matrix[i][j] == score:
                traceback_matrix[i][j] = 1
            elif score_matrix[i][j] == score_matrix[i - 1][j] + gap_cost:
                traceback_matrix[i][j] = 2
            elif score_matrix[i][j] == score_matrix[i][j - 1] + gap_cost:
                traceback_matrix[i][j] = 3

            if score_matrix[i][j] >= max_score:
                max_score = score_matrix[i][j]
                max_pos = (i, j)

    # Traceback to find the best alignment
    align1 = []
    align2 = []
    i, j = max_pos

    while traceback_matrix[i][j] != 0:
        if traceback_matrix[i][j] == 1:
            align1.append(seq1[i - 1])
            align2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif traceback_matrix[i][j] == 2:
            align1.append(seq1[i - 1])
            align2.append('-')
            i -= 1
        elif traceback_matrix[i][j] == 3:
            align1.append('-')
            align2.append(seq2[j - 1])
            j -= 1

    align1.reverse()
    align2.reverse()

    return max_score, align1, align2, score_matrix, traceback_matrix


import numpy as np


def smith_watermanV2(seq1, seq2, match_score=2, gap_cost=-1, mismatch_cost=-1):
    """
    Perform local alignment between two sequences using the Smith-Waterman algorithm.

    :param seq1: First sequence (list of subwords in Arabic).
    :param seq2: Second sequence (list of subwords in Arabic).
    :param match_score: Score for matching characters.
    :param gap_cost: Cost (negative) for introducing a gap.
    :param mismatch_cost: Cost (negative) for mismatching characters.
    :return: The optimal local alignment score, the alignments, the score matrix, the traceback matrix,
             the direction matrix, and the matching matrix.
    """
    # Initialize the scoring, traceback, direction, and matching matrices
    n = len(seq1)
    m = len(seq2)
    score_matrix = np.zeros((n + 1, m + 1))
    traceback_matrix = np.zeros((n + 1, m + 1), dtype=int)
    direction_matrix = np.zeros((n + 1, m + 1, 3), dtype=int)  # One-hot encoding for directions
    matching_matrix = np.zeros((n + 1, m + 1), dtype=int)  # 1 for match, 0 for mismatch

    # Fill the scoring, traceback, direction, and matching matrices
    max_score = 0
    max_pos = (0, 0)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                score = score_matrix[i - 1][j - 1] + match_score
                matching_matrix[i][j] = 1  # Match
            else:
                score = score_matrix[i - 1][j - 1] + mismatch_cost
                matching_matrix[i][j] = 0  # Mismatch

            score_matrix[i][j] = max(0, score,
                                     score_matrix[i - 1][j] + gap_cost,
                                     score_matrix[i][j - 1] + gap_cost)

            if score_matrix[i][j] == 0:
                traceback_matrix[i][j] = 0  # End of alignment
                direction_matrix[i][j] = [0, 0, 0]  # No direction
            elif score_matrix[i][j] == score:
                traceback_matrix[i][j] = 1  # Diagonal
                direction_matrix[i][j] = [1, 0, 0]  # One-hot for diagonal
            elif score_matrix[i][j] == score_matrix[i - 1][j] + gap_cost:
                traceback_matrix[i][j] = 2  # Up
                direction_matrix[i][j] = [0, 0, 1]  # One-hot for down
            elif score_matrix[i][j] == score_matrix[i][j - 1] + gap_cost:
                traceback_matrix[i][j] = 3  # Left
                direction_matrix[i][j] = [0, 1, 0]  # One-hot for left

            if score_matrix[i][j] >= max_score:
                max_score = score_matrix[i][j]
                max_pos = (i, j)

    # Traceback to find the best alignment
    align1 = []
    align2 = []
    i, j = max_pos

    while traceback_matrix[i][j] != 0:
        if traceback_matrix[i][j] == 1:
            align1.append(seq1[i - 1])
            align2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif traceback_matrix[i][j] == 2:
            align1.append(seq1[i - 1])
            align2.append('-')
            i -= 1
        elif traceback_matrix[i][j] == 3:
            align1.append('-')
            align2.append(seq2[j - 1])
            j -= 1

    align1.reverse()
    align2.reverse()

    return max_score, align1, align2, score_matrix, traceback_matrix, direction_matrix, matching_matrix

# def smooth_smith_waterman:
#     pass
# if __name__ == "__main__":
# # Example usage
# seq1 = ["A", "G", "T", "C"]
# seq2 = ["G", "T", "A", "C"]
#
# max_score, align1, align2, score_matrix, traceback_matrix, direction_matrix, matching_matrix = smith_watermanV2(seq1, seq2)
#
# print("Max Score:", max_score)
# print("Alignment 1:", align1)
# print("Alignment 2:", align2)
# print("Score Matrix:\n", score_matrix)
# print("Traceback Matrix:\n", traceback_matrix)
# print("Direction Matrix:\n", direction_matrix)
# print("Matching Matrix:\n", matching_matrix)


# Example usage

# #
# if __name__ == "__main__":
#
#     # for i in range(0, 5):
#     _, sentencewithNwords = fetch_arabic_sentence(n=5)
#     subWordsOfNwords = tokenize_based_on_non_connecting_letters(sentencewithNwords)
#     _, word = fetch_arabic_sentence(n=3)
#     subWordsOfAddwords = tokenize_based_on_non_connecting_letters(word)
#     print("Fetched Arabic Sentence:", subWordsOfNwords)
#     print("Fetched Arabic Word:", subWordsOfAddwords)
#     originalSubWords = subWordsOfNwords.copy()
#     addAugmetation = augment_sentence(subWordsOfNwords, new_words=subWordsOfAddwords, operation="add")
#     print(" After Add Augmentation Arabic Sentence:", addAugmetation)

#     print("originalSubWords",originalSubWords)
#     print("addAugmetation",addAugmetation)
#     score, alignment1, alignment2,score_matrix,track_matrix= smith_waterman(originalSubWords, addAugmetation)
#     print("Alignment Score:", score)
#     print("Alignment 1:", alignment1)
#     print("Alignment 2:", alignment2)
#     print(track_matrix)
#     plt.imshow(score_matrix)
#     plt.show()

#
# deleteAugmetation = augment_sentence(subWordsOfNwords, new_words=subWordsOfAddwords, num_to_delete=1,
#                                      operation="delete")
# print(" After Del Augmentation Arabic Sentence:", deleteAugmetation)
# score, alignment1, alignment2 = smith_waterman(subWordsOfNwords, deleteAugmetation)
# print("Alignment Score:", score)
# print("Alignment 1:", alignment1)
# print("Alignment 2:", alignment2)

# exit(0)
#
