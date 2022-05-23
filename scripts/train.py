import logging

from gensim.models import FastText

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