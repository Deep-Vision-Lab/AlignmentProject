

from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import matplotlib.pylab as plt
from fetch_arabic_sentence import fetch_arabic_sentence
import random
from splitTextLine import tokenize_based_on_non_connecting_letters


def add_words_or_components(components, new_words):
    """
    Add a list of new words or connected components at random positions in the components list.
    """
    for word in new_words:
        insert_position = random.randint(0, len(components))
        components.insert(insert_position, word)
    return components


def delete_words_or_components(components, num_to_delete):
    """
    Delete a random number of words or connected components from the components list.
    """
    for _ in range(min(num_to_delete, len(components))):
        delete_position = random.randint(0, len(components) - 1)
        components.pop(delete_position)
    return components


def augment_sentence(components, new_words=None, num_to_delete=0, operation="add"):
    """
    Augment the sentence by adding or deleting a list of words, letters, or connected components.

    :param sentence: The connected component sentence to be augmented.
    :param new_words: The list of new words or components to add if the operation is 'add'.
    :param num_to_delete: The number of words or components to delete if the operation is 'delete'.
    :param operation: The type of augmentation to perform ('add' or 'delete').
    :return: The augmented sentence.
    """

    ComponentsToModify = components.copy()
    # Split the sentence into components
    if operation == "add" and new_words:
        ComponentsToModify = add_words_or_components(ComponentsToModify, new_words)
    elif operation == "delete" and num_to_delete > 0:
        ComponentsToModify = delete_words_or_components(ComponentsToModify, num_to_delete)

    # Reconstruct the sentence
    # augmented_sentence = ''.join(components)
    return ComponentsToModify



if __name__ == '__main__':

    _, sentencewithNwords = fetch_arabic_sentence(n=7)
    subWordsOfNwords = tokenize_based_on_non_connecting_letters(sentencewithNwords)
    _, word = fetch_arabic_sentence(n=2)
    subWordsOfAddwords = tokenize_based_on_non_connecting_letters(word)
    print("Fetched Arabic Sentence:", subWordsOfNwords)
    print("Fetched Arabic Word:",subWordsOfAddwords)
    addAugmetation = augment_sentence(subWordsOfNwords, new_words=subWordsOfAddwords, operation="add")
    print(" After Add Augmentation Arabic Sentence:", addAugmetation)

    deleteAugmetation = augment_sentence(subWordsOfNwords, new_words=subWordsOfAddwords, num_to_delete=1,
                                         operation="delete")
    print(" After Del Augmentation Arabic Sentence:", deleteAugmetation)

