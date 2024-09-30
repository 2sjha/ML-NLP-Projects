"""
Shubham Shekhar Jha (sxj220028)
Homework 1: Word Guess Game
"""

from random import seed, randint
from sys import argv, stderr
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag


def main():
    """
    Driver function to perform HW1 tasks
    """
    # Check if argument is provided to our python script
    if len(argv) < 2:
        print("Error: No argument provided.", file=stderr)
        return

    # Read the text in the file provided as argument
    filename = argv[1]
    raw_text = ""
    with open(filename, encoding="utf-8") as file:
        raw_text = file.read()

    # Tokenize the raw text and calculate lexical diversity
    tokens = word_tokenize(raw_text)
    unique_tokens = set(tokens)
    lexical_diversity = len(unique_tokens) / len(tokens)
    lexical_diversity = round(lexical_diversity, 2)
    print("(2) Lexical Diversity of", filename, "is", lexical_diversity)

    # Preprocess the raw text to get filtered tokens & nouns
    (tokens, nouns) = preprocess_text(raw_text)

    # Create a sorted dictionary of nouns with their counts in descending order
    nouns_count = {noun: tokens.count(noun) for noun in nouns}
    nouns_count = dict(
        sorted(nouns_count.items(), key=lambda item: item[1], reverse=True)
    )

    # Fetch Top 50 most common nouns for the Guessing game
    top_50_nouns = {noun: nouns_count[noun] for noun in list(nouns_count.keys())[:50]}
    print("(4) 50 Most common words and their count: ", top_50_nouns)
    game_words = list(top_50_nouns.keys())

    # Start the Guessing Game with Top 50 Nouns
    guessing_game(game_words)


def preprocess_text(raw_text):
    """
    Processes the text to extract the tokens and nouns from the raw_text
    """
    # Tokenize the text and select only those tokens
    # that are NOT in NLTK stopwords
    # and that contain alphabetical characters ONLY
    # and that are longer than 5 characters
    tokens = word_tokenize(raw_text)
    stop_words = set(stopwords.words("english"))
    tokens = [
        token.lower()
        for token in tokens
        if token not in stop_words and token.isalpha() and len(token) > 5
    ]

    # Lemmatize the tokens to get a set of unique lemmas and tag their parts of speech
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    lemmas = set(lemmas)
    pos_tagged_words = pos_tag(lemmas)
    print("(3-c) First 20 words with POS tags:", pos_tagged_words[:20])

    # Extract the nouns (excluding Proper Nouns "NNP" and "NNPS") from the pos_tagged words list
    nouns = [
        word_tag[0]
        for word_tag in pos_tagged_words
        if word_tag[1] == "NN" or word_tag[1] == "NNS"
    ]
    print("(3-e) Total No. of tokens:", len(tokens), ", and No. of Nouns: ", len(nouns))
    return (tokens, nouns)


def guessing_game(words):
    """Simulates the Guessing Game"""
    # Initialize game with random word and score = 5
    score = 5
    guess = ""
    seed(10000)
    word = words[randint(0, len(words) - 1)]
    wordchars = set(list(word))  # Character set of the word
    guesses = set()  # Keeps correct guesses so far
    print("\n\nLet's play a word guessing game!")

    # Play until score becomes negative
    while score >= 0:
        # Print Guess space and wait for user input
        print_guess(word, guesses)
        guess = input("Guess a letter: ")

        # Restart loop ip the user input is invalid
        if len(guess) != 1:
            print("Invalid guess.")
            continue

        # guessing ! means that the user wishes to end the game
        if guess == "!":
            print("Ending the Game! Final Score:", score)
            break

        # If the Guess is correct, add it to the correct guesses charset and increase score
        if guess in guesses:
            print("You've already guessed", guess + ".", "Guess again.")
        elif guess in wordchars:
            guesses.add(guess)
            score += 1
            print("Right! Score is", score)
        else:
            score -= 1
            print("Sorry, guess again. Score is", score)

        # size of word charset == size of correct guesses means the word is complete.
        if len(guesses) == len(wordchars):
            print_guess(word, guesses)
            print("You solved it!\n")
            print("Current Score: ", score, "\n")

            # Restart Game by choosing a random word
            print("Guess another word")
            word = words[randint(0, len(words) - 1)]
            wordchars = set(list(word))
            guesses = set()

    if score < 0:
        print("Negative score. The word is", word + ".", "Ending the Game!")


def print_guess(word, guesses):
    """
    Prints the Guessing underscores
    """
    for c in word:
        if c in guesses:
            print(c, end=" ")
        else:
            print("_", end=" ")
    print("")


if __name__ == "__main__":
    main()
