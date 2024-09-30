"""
Script to clean the scraped files
"""

from typing import Set
from utils import check_contains
from filters import DISCARD_WORDS_LIST, NUM_CLEAN_FILES
from nltk.tokenize import sent_tokenize


def clean_crawled_text(text: str) -> Set[str]:
    """
    Cleans text from one crawled link into list of relevant & clean lines.
    """
    # Create set of lines because some crawled text files
    # have duplicate lines in the text from parsed html
    lines_st = set()
    lines = []
    sentences = sent_tokenize(text)
    for line in sentences:
        line = line.strip()
        if line and line not in lines_st:
            lines_st.add(line)
            lines.append(line)

    # Only keep lines that dont have DISCARD_WORDS and lines that have more than 10 words.
    cleaned_lines = []
    for line in lines:
        if not check_contains(line, DISCARD_WORDS_LIST) and len(line.split(" ")) > 10:
            cleaned_lines.append(line)

    return cleaned_lines


def clean_files():
    """
    Reads in the crawled files and cleans them to be processed for the Knowledge Base
    """
    # Read all the crawled files text and clean them individually
    cleaned_text = {}
    for i in range(0, NUM_CLEAN_FILES):
        with open("crawled_files/crawl_" + str(i) + ".txt", encoding="utf8") as f:
            text = f.read()
            cleaned_text[i] = clean_crawled_text(text)

    # Save cleaned files to disk
    for idx, clean_page_data in cleaned_text.items():
        with open(
            "cleaned_files/clean_" + str(idx) + ".txt", "w", encoding="utf8"
        ) as f:
            for line in clean_page_data:
                f.write(line + "\n")


if __name__ == "__main__":
    clean_files()
