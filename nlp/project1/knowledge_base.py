"""
Script to create a knowledge base for terms from a corpus
"""

from typing import List, Dict
import pickle
from filters import NUM_CLEAN_FILES, IMPORTANT_TERMS
from nltk.tokenize import sent_tokenize
from utils import check_contains, save_dict_as_txt


def create_knowledge_base(imp_terms: List[str]) -> Dict[str, List[str]]:
    """
    Creates Knowledge Base for the important terms using the text
    """

    # Knowledge Base is a dictionary of terms, list[revelant text]
    knowledge_base = {}

    for i in range(0, NUM_CLEAN_FILES):
        with open("cleaned_files/clean_" + str(i) + ".txt", encoding="utf8") as f:
            text = f.read()
            sentences = sent_tokenize(text)

            for term in imp_terms:
                term = term.lower()
                term_parts = term.split(" ")
                for i, sentence in enumerate(sentences):

                    if "\n" not in sentence:
                        # If the term is found, then the sentence might be relevant
                        if check_contains(sentence.lower(), term_parts):

                            if term in knowledge_base:
                                knowledge_base[term].append(sentence)
                            else:
                                knowledge_base[term] = [sentence]
                    else:
                        # Handle the case where sentence tokenizer groups multiple sentences together
                        sent_parts = sentence.split("\n")
                        for sent_pt in sent_parts:
                        # If the term is found, then the sentence might be relevant
                            if check_contains(sent_pt.lower(), term_parts):
                                if term in knowledge_base:
                                    knowledge_base[term].append(sent_pt)
                                else:
                                    knowledge_base[term] = [sent_pt]

    return knowledge_base


def save_knowledge_base(knowledge_base: Dict[str, List[str]], kb_format: str):
    """
    Saves Knowledge Base as a text file
    """
    if kb_format != "txt" and kb_format != "pickle":
        print("Invalid format to save Knowledge Base")
        return

    if kb_format == "txt":
        save_dict_as_txt(knowledge_base, "knowledge_base.txt")
    else:
        with open("knowledge_base.pkl", "wb") as f:
            pickle.dump(knowledge_base, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    kb = create_knowledge_base(IMPORTANT_TERMS)
    save_knowledge_base(kb, "txt")
    save_knowledge_base(kb, "pickle")
