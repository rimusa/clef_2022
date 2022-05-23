import logging
import sys

from train import train_embeddings
from utils import clean, get_counts
from files import iter_corpus



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
    iter_corpus(texts_path, corpus, several_dirs=False)
        
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
                    

    
    