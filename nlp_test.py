

#%% 
import pandas as pd
#%% 
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
print(script_dir)

#%% 

url = "https://raw.githubusercontent.com/clairett/pytorch-sentiment-classification/master/data/SST2/train.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label','text'])

df.head()

#0 = negative sentiment, 1=positive sentiment

#%% 