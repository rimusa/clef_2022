import logging
import torch
import json
import os
        
        
def iter_corpus(texts_path, corpus, several_dirs=False):
    if several_dirs:
        paths = os.listdir(texts_path)
        for path in paths:
            generate_corpus(texts_path+path, corpus)
    else:
        generate_corpus(texts_path, corpus)
        
        
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
                with open(path+file, "r") as F:
                    news = json.load(F)
                    for article in news:
                        title = clean(article["title"])
                        if title != "":
                            G.write(title + "\n")
                        body = clean(article["content"])
                        if title != "":
                            G.write(body + "\n")
                            
                            

def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)