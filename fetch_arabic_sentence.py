import os
import random

##
'''
:
This extended function allows you to not only fetch a random 
sentence from a randomly selected file within one of the datasets 
but also to extract a specified number of random words from that sentence. 
This makes the function more versatile, depending on your specific needs for
 working with Arabic text.

'''
def fetch_arabic_sentence(n,datasets=["datasets/ArabicDialect", "datasets/QuranDataset"]):
    """
    Fetch a random Arabic sentence from a randomly chosen text file within one of the datasets.

    :param datasets: List of dataset directories to choose from.
    :return: A randomly chosen Arabic sentence from the file.
    """
    # Step 1: Pick one of the datasets
    dataset = random.choice(datasets)

    # Step 2: Pick a random text file from the chosen dataset
    dataset_path = os.path.join(os.getcwd(), dataset)
    if not os.path.exists(dataset_path):
        return None, f"The dataset path {dataset_path} does not exist."

    txt_files = [f for f in os.listdir(dataset_path) if f.endswith('.txt')]

    if not txt_files:
        return None, f"No text files found in the dataset: {dataset}."

    chosen_file = random.choice(txt_files)
    file_path = os.path.join(dataset_path, chosen_file)

    # Step 3: Open the file and read lines
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    if not lines:
        return None, f"The file {chosen_file} is empty."

    # Step 4: Pick a random line from the file
    random_line = random.choice(lines).strip()

    # Step 5: Split the line into words
    words = random_line.split()

    if len(words) < n:
        selected_words = " ".join(words)
        return random_line, selected_words
        print("FIX IT YOU SHOULD GET AT LEAST N !!!!")

    # Step 6: Randomly select `n` words from the list of words
    selected_words = random.sample(words, n)
    selected_words = " ".join(selected_words)
    return random_line, selected_words

#
# # Example usage
# if __name__ == "__main__":
#     for i in range(0,100):
#         arabic_sentence = fetch_arabic_sentence(n=7)
#         print("Fetched Arabic Sentence:", arabic_sentence)