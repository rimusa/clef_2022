import logging
import pickle
import nltk
import json
import sys
import os

from gensim.models import FastText
from nltk.tokenize import word_tokenize

nltk.download('punkt')

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
        
        
def generate_corpus(path, corpus):
    """
    Generates a corpus from a series
    """
    files = os.listdir(path)

    total = len(files)
    counter = 0

    logging.info("Found " + str(total) + " news sources")

    with open(corpus, "a+") as G:
        for file in files:
            counter += 1
            if (counter % 10 == 0) or (counter == total):
                logging.info(str(counter) + " of " + str(total) + " sources parsed...")
            if file != ".empty":
                with open(path+"/"+file, "r") as F:
                    news = json.load(F)
                    for article in news:
                        title = clean(article["title"])
                        if title != "":
                            G.write(title + "\n")
                        body = clean(article["content"])
                        if title != "":
                            G.write(body + "\n")
                        
                        
def train_embeddings(corpus, model_path):
    """
    Trains and saves a FastText model from a corpus file
    """
    
    logging.info("Initializing embedding model...")
    nela_model = FastText(vector_size=300, sg=1)
    logging.info("Building vocabulary...")
    nela_model.build_vocab(corpus_file=corpus)
    logging.info("Begin training!")
    nela_model.train(
                     corpus_file=corpus, epochs=nela_model.epochs,
                     total_examples=nela_model.corpus_count, total_words=nela_model.corpus_total_words
                    )
    logging.info("Training successfully finished!")
    logging.info("Saving model...")
    nela_model.save(model_path)
    logging.info("Model saved!")


def main(texts_path, results_path, emb_path, emb_name, several_dirs=False):           
    wordlist = results_path + emb_name + "_wordcount.pkl"
    corpus = results_path + emb_name + "_texts.txt"
    model_path = emb_path + emb_name
    
    logging.info("Checking whether the corpus file exists")
    try:
        f = open(corpus, "x")
        f.close()
        logging.info("Corpus file created!")
    except FileExistsError:
        f = open(corpus, "w")
        f.close()
        logging.info("Corpus file reset!")
           
    logging.info("Creating the embeddings corpus...")
    if several_dirs:
        paths = os.listdir(texts_path)
        for path in paths:
            generate_corpus(texts_path+path, corpus)
    else:
        generate_corpus(texts_path, corpus)
        
    logging.info("Embeddings corpus created!")
        
    logging.info("Creating the vocabulary file...")
    get_counts(corpus, wordlist)
    
    logging.info("Getting ready to train the embeddings!")
    train_embeddings(corpus, model_path)
    logging.info("Embeddings trained! :)")
    
    

if __name__ == "__main__":

    logging.basicConfig(format='%(levelname)s %(asctime)s %(message)s',
                        filename='logs/news_embeddings.log', 
                        encoding='utf-8', 
                        level=logging.DEBUG)

    texts_path = sys.argv[1]
    results_path = sys.argv[2]
    emb_path = sys.argv[3]
    emb_name = sys.argv[4]

    main(texts_path, results_path, emb_path, emb_name, several_dirs=True)
                    

    
    