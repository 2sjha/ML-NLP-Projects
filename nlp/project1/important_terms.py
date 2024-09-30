"""
Script to find important terms from a corpus
"""

from string import punctuation
from math import log
from utils import term_frequency
from filters import NUM_CLEAN_FILES
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy


def important_terms(num_terms: int):
    """
    Finds important terms using tf-idf from cleaned text files
    """
    stop_words = set(stopwords.words("english"))
    spacy_nlp = spacy.load("en_core_web_md")
    file_text = [0] * NUM_CLEAN_FILES
    terms = {}
    for i in range(0, NUM_CLEAN_FILES):
        with open("cleaned_files/clean_" + str(i) + ".txt", encoding="utf8") as f:
            text = f.read()
            tokens = word_tokenize(text)
            tokens = [
                token.lower()
                for token in tokens
                if token not in stop_words
                and token not in punctuation
                and len(token) > 5
            ]
            file_text[i] = text

            # Put all NER terms into important terms
            # Start with tf = 1 and df = 1 to avoid division by zero
            spacy_text = spacy_nlp(text)
            for ent in spacy_text.ents:
                terms[ent.text.lower()] = [0, 0, 0]

            # Find one word tokens in this document that have frequency greater than 3
            frequent_tokens = {}
            for token in tokens:
                if token in frequent_tokens:
                    frequent_tokens[token] += 1
                else:
                    frequent_tokens[token] = 1

            # Put frequent one word tokens into important terms
            # Start with tf-df = 1,1
            for token, frequency in frequent_tokens.items():
                if frequency > 3:
                    terms[token] = [0, 0, 0]

    # Calculate Term Frequency and Document Frequency for each term
    for i in range(0, NUM_CLEAN_FILES):
        text = file_text[i][0]
        for term, tf_df in terms.items():
            if term in text:
                tf_df[0] += term_frequency(term, text)
                tf_df[1] += 1

    # Calculate Tf-IDf score for each term
    for term, metrics in terms.items():
        tf = metrics[0]
        df = metrics[1]
        if tf and df:
            tf_idf = tf * log(1 + NUM_CLEAN_FILES / df)
            metrics[2] = tf_idf

    # Choose the top num_terms terms by their tf-idf scores
    terms = dict(sorted(terms.items(), key=lambda x: x[1][2], reverse=True)[:num_terms])

    # print(terms)
    print(terms.keys())


if __name__ == "__main__":
    important_terms(50)
