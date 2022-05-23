from gensim.models import FastText

import pandas as pd
import numpy as np

from nltk.tokenize import word_tokenize
nltk.download('punkt')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import argparse
import math

from train import train, evaluate
from utils import preprocess, one_hot, labeling
from files import save





class CNN_Text(nn.Module):
    
    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args
        
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        
        self.embed = args.embeddings
        
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, C)

    def forward(self, x):
        batch_size = x.map(len).max()
        x_embeds = torch.zeros((len(x),batch_size,300))
        for i in range(len(x)):
            A = torch.FloatTensor(self.embed.wv[x.iloc[i]])
            x_embeds[i,:A.shape[0],:] = A
        x = x_embeds
    
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N, len(Ks)*Co)
        
        logit = self.fc1(x)  # (N, C)
        return logit


def read_arguments():
    parser = argparse.ArgumentParser()

    
    args = parser#.parse_args()
    
    args.save_best=True
    args.save_dir="./cnn/snapshot"
    
    args.dropout = 0.5
    args.max_norm=3.0
    args.embed_dim=300
    args.kernel_num=100
    args.kernel_sizes=[3,4,5]
    args.no_cuda=False
    
    args.early_stop=200
    args.save_dir='./cnn/snapshot'
    args.save_interval=500
    args.test_inteval=20
    args.log_inteval=15
    args.batch_size=64
    args.epochs=256
    args.lr=0.001
    
    return args





train_path = "./Task3_2022/Task3_train_dev/Task3_english_training.csv"
dev_path = "./Task3_2022/Task3_train_dev/Task3_english_dev.csv"
embeddings_path = "./fasttext_nela_2021.bin"

train_df = preprocess(train_path, shear=True)
dev_df   = preprocess(dev_path, shear=True)

labels = train_df["label"].unique()
rating_dict = dict(zip(labels, range(len(labels))))

args = read_arguments()
args.class_num = len(labels)
args.embeddings = FastText.load(embeddings_path)

args.cuda = (not args.no_cuda) and torch.cuda.is_available()
if args.cuda:
        torch.cuda.set_device(args.device)
        cnn = cnn.cuda()
        
cnn = CNN_Text(args)
train(train_df,dev_df,cnn,args)


model_loaded = CNN_Text(args)
model_loaded.load_state_dict(torch.load("./cnn/snapshot/best_steps_model.pt"))
model_loaded.evaluate()


test_path = "./Task3_2022/Task3_Test/English_data_test_release.csv"
test_df = preprocess(test_path, shear=False)


reverse_rating = {
    rating_dict[i] : i for i in rating_dict
}


labels = []
for i in test_df.iterrows():
    data = i[1]
    ID = data.public_id
    text = [data.title_token]
    
    
    if (type(text[0])==float and math.isnan(text[0])) or (len(text[0]) < 5):
        current_label = "other"
        
    else:
        text = pd.Series(text)
        logit = model_loaded(text)
        prediction = torch.max(logit, 1)[1]
        current_label = reverse_rating[prediction.item()]
    #print(current_label)
    
    labels.append({"public_id":ID, "prediction":current_label})
    
labels = pd.DataFrame(labels)

labels.to_csv("subtask3_english_sprakbankenteam.tsv", sep="\t", index=False)