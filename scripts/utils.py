import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

import logging
import pickle

import pandas as pd

from collections import defaultdict


def clean(text):
    """
    Tokenizes text using word_tokenize from nltk
    """
    T = [token for token in word_tokenize(text) if token not in ["@", "#"]]
    line = " ".join(T)
    return line


def get_counts(corpus, wordlist):
    """
    Generates a count dictionary given a corpus text file
    """
    counts = defaultdict(int)
    counter = 0
    logging.info("Generating word counts")
    with open(corpus, "r") as F:
        for line in F:
            counter += 1
            if counter % 100000 == 0:
                logging.info(str(counter) + " lines counted...")
            tokens = line.split()
            for token in tokens:
                counts[token.lower()] += 1
    
    logging.info(str(counter) + " total lines counted.")
    
    with open(wordlist, "wb") as F:
        pickle.dump(counts, F)
        
    logging.info("Wordcount dictionary saved at " + wordcount)
    
    

def preprocess(path, shear=False):
    df = pd.read_csv(path)

    df["label"] = df["our rating"].str.lower()
    df["title"] = df["title"].str.lower()

    df["title_token"] = df["title"].map(word_tokenize, na_action='ignore')

    if shear:
        df["title_len"] = df["title_token"].map(len, na_action='ignore')
        df = df[df.title_len>4]
        df["title_log"] = df["title_len"].map(np.log)
        df = df[df.title_log < 4]
    
    return df


def one_hot(rating, n_labels, label_dict, default="other"):
    out = [0 for _ in range(n_labels)]
    try:
        idx = label_dict[rating]
    except KeyError:
        idx = label_dict[default]
    out[idx] = 1
    return out


def labeling(targets, n_labels, label_dict, default="other"):
    out = []
    for target in targets:
        out.append(one_hot(target, n_labels=len(labels), label_dict=rating_dict, default="other"))
    return out