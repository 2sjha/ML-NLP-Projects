"""
Utility functions
"""

from typing import List, Dict, Tuple, Any
import pandas as pd
import pickle


NUM_FILES = 29

IMPORTANT_TERMS = [
    "Office",
    "Dunder Mifflin",
    "Scranton",
    "Pennsylvania",
    "Ryan Howard",
    "Michael Scott",
    "Meredith Palmer",
    "Christmas",
    "Carol Stills",
    "Jan Levinson",
    "Dwight Schrute",
    "Angela Martin",
    "Jim Halpert",
    "Pam Beesly",
    "Roy Anderson",
    "Stamford",
    "Oscar Martinez",
    "Andy Bernard",
    "Karen Filippelli",
    "Phyllis Lapin",
    "Bob Vance",
    "David Wallace",
    "Toby Flenderson",
]

PEOPLE = [
    "Ryan Howard",
    "Michael Scott",
    "Meredith Palmer",
    "Carol Stills",
    "Jan Levinson",
    "Dwight Schrute",
    "Angela Martin",
    "Jim Halpert",
    "Pam Beesly",
    "Roy Anderson",
    "Oscar Martinez",
    "Andy Bernard",
    "Karen Filippelli",
    "Phyllis Lapin",
    "Bob Vance",
    "David Wallace",
    "Toby Flenderson",
]

OTHERS = [
    "The Office (US)",
    "Dunder Mifflin",
    "Scranton",
    "Pennsylvania",
    "Christmas",
    "Stamford",
]


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


def check_contains(line: str, str_list: List[str]) -> bool:
    """
    Returns True if line contains any string from str_list
    """
    for string in str_list:
        if string in line:
            return True
    return False


def load_pickle(pickle_name: str) -> Dict[Any, any]:
    """
    Loads a pickled file into a python container
    In our usecase it'll be the knowledge base dictionary
    """
    with open(pickle_name, "rb") as handle:
        pickle_dict = pickle.load(handle)

    return pickle_dict


def load_qa_dataset() -> pd.DataFrame:
    """
    Load the custom QnA dataset
    """
    questions = []
    answers = []
    for i in range(NUM_FILES):
        with open("qa/qa_" + str(i) + ".txt") as f:
            qa_data = f.read()
            q, a = read_qa(qa_data)
            questions += q
            answers += a
    return pd.DataFrame({"question": questions, "answer": answers})


def read_qa(qa_data: str) -> Tuple[List[str], List[str]]:
    """
    Parse the QnA text into QnA lists
    """
    q = []
    a = []
    for line in qa_data.split("\n"):
        line = line.strip()
        if not line:
            continue
        elif line.startswith("Q: "):
            q.append(line[3:])
        elif line.startswith("A: "):
            ans = "<START> " + line[3:] + " <END>"
            a.append(ans)
    return q, a
