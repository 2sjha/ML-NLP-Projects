"""
Utility functions
"""

from typing import List, Set, Dict, Any
from re import Pattern, search
from nltk import ngrams, word_tokenize


class Page:
    """
    Represents a link to be be crawled
    Contains the URL of the Page,
    the divs that need to be read in that page (manually picked for initial links
    & empty for crawled links),
    """

    def __init__(self, url: str, crawl_links: bool):
        self.url = url
        self.crawl_links = crawl_links


class CrawledData:
    """
    Data from a crawled page
    Contains text from the page,
    more links that the page contains
    """

    def __init__(self, url: str, text: str, crawled_links: Set[str]):
        self.url = url
        self.text = text
        self.crawled_links = crawled_links


def get_domain(url: str) -> str:
    """
    Returns the domain from a URL
    """
    parts = url.split("/")
    domain = "/".join(parts[:3])
    return domain


def match(string: str, str_patterns: Set[Pattern]) -> bool:
    """
    Returns True if url matches with any url pattern
    """
    for str_pattern in str_patterns:
        matched = search(str_pattern, string)
        if matched:
            return True
    return False


def check_contains(line: str, str_list: List[str]) -> bool:
    """
    Returns True if line contains any string from str_list
    """
    for string in str_list:
        if string in line:
            return True
    return False


def term_frequency(term: str, text: str) -> float:
    """
    Returns the frequency of term in text
    """
    count = 0
    n = term.count(" ") + 1
    text_tokens = word_tokenize(text)

    if n == 1:
        count = text_tokens.count(term)
    else:
        text_ngrams = list(ngrams(text_tokens, n))
        for ngram in text_ngrams:
            if term in " ".join(ngram):
                count += 1

    return count / len(text_tokens)


def save_dict_as_txt(dictionary: Dict[Any, List[Any]], file_name: str):
    """
    Saves a dictionary in a txt file with custom formatting for easier editing
    """
    with open(file_name, "w", encoding="utf8") as f:
        for term, info_items in dictionary.items():
            f.write(term + ": {\n")
            for info in info_items:
                f.write("\t" + info + "\n\n")
            f.write("}\n\n")
