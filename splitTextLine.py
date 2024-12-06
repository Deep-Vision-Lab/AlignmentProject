

from fetch_arabic_sentence import fetch_arabic_sentence
'''
generates an image with that sentence in white text on a black background.
 The font size and padding are customizable, making it adaptable
 to different text lengths and formats.
'''

def tokenize_based_on_non_connecting_letters(text):
    # Expanded list of non-connecting Arabic characters
    non_connecting_letters = {'ا', 'د', 'ذ', 'ر', 'ز', 'و', 'ى', ' ', 'أ', 'إ', 'ؤ', 'ء'}

    # Tokenize based on the presence of non-connecting letters or spaces
    tokens = []
    current_token = ''

    for char in text:
        current_token += char

        if char in non_connecting_letters:
            tokens.append(current_token)
            current_token = ''

    # Add the last token if it exists
    if current_token:
        tokens.append(current_token)

    # strip all componets with
    lst=[]
    for t in tokens:
        if ' ' in t and len(t)> 1:
           lst.append(t.strip())
        else:
            lst.append(t)
    lst = remove_whitespace_from_subwords(lst)
    return lst

def remove_whitespace_from_subwords(subwords_list):
    """
    Remove all empty strings or strings consisting solely of white spaces from the list of subwords.

    :param subwords_list: List of subwords which may include spaces or white spaces.
    :return: List of subwords with all white spaces removed.
    """
    # Use list comprehension to filter out empty or whitespace-only subwords
    cleaned_subwords_list = [subword for subword in subwords_list if subword.strip()]
    return cleaned_subwords_list


# if __name__ == '__main__':
#
#     arabic_sentence = fetch_arabic_sentence(n=10)
#     fullSentence,sentencewithNwords = arabic_sentence
#     print("Fetched Arabic Sentence:", arabic_sentence)
#     print("Fetched fullSentence Sentence:", fullSentence)
#     print("Fetched sentencewithNwords Sentence:", sentencewithNwords)
#     splitSubwords = tokenize_based_on_non_connecting_letters(sentencewithNwords)
#     print("The splited subword:",splitSubwords)