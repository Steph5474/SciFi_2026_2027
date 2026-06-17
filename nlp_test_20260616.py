

#%% 
import pandas as pd
#%% 
import os
print(os.getcwd())

script_dir = os.path.dirname(os.path.abspath(__file__))
print(script_dir)

#%% 
#it seems the data is a movie review/comment about a 1933 <<movie Les vingt-huit jours de Clairette>>
url = "https://raw.githubusercontent.com/clairett/pytorch-sentiment-classification/master/data/SST2/train.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['text','label'])
df.head()
#label: 0 = negative sentiment, 1=positive sentiment. This is human-annotated. 

print(df.shape)  
#6920,2 
#%%
print(df['label'].value_counts(dropna=False))

"""
label
1    3610
0    3310
Name: count, dtype: int64
"""

#%% 
df.head(500).to_excel("clairett_sentiment_data_head500.xlsx", index=False)

# %%
"""  now apply VADER sentiment analysis (fast + easy, generate scores -1 to 1)"""


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

#VADER generates 4 columns: neg, neu, pos, compound
#compound is the overall sentiment score from -1 (most negative) to 1 (most positive)
#neg/neu/pos - they always sum up to 1 (proportions of the text being negative, neutral, or positive) 
#use compound for the overall sentiment score, use neg/neu/pos for the individual sentiment components
    

df["vader_neg"] = df["text"].apply(lambda x: analyzer.polarity_scores(x)["neg"])
df["vader_neu"] = df["text"].apply(lambda x: analyzer.polarity_scores(x)["neu"])
df["vader_pos"] = df["text"].apply(lambda x: analyzer.polarity_scores(x)["pos"])
df["vader_compound"] = df["text"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
print(df.head()) 
      

# %%
""" now check if sentiment scores agree the human-annotated labels 

there are several ways to check this:

1. Compare the distribution of VADER scores for each label. Whether there is a clear separation between the scores for different labels.
   For example, for those labeled as "negative", we expect the VADER scores to be mostly negative.
    and for those labeled as "positive", we expect the VADER scores to be mostly positive. 
   
2. Classification agreement.
  This involves comparing the predicted sentiment labels with the actual human-annotated labels to see how often they match.
  Need to set a threshold for classification (e.g., if VADER compound score > 0.05, classify as positive, otherwise negative).
  Then look at the confusion matrix to see how often the predictions match the labels.

3. Calculate the correlation between VADER scores and human labels
   Or calculate chi-square test to see if the scores and labels are independent. 

"""

#%% approach 1: check the distribution of VADER scores for each label
neg_scores = df[df["label"] == 0]["vader_compound"]
pos_scores = df[df["label"] == 1]["vader_compound"]

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

plt.hist(neg_scores, bins=30, alpha=0.6, color='red', label='Human label = 0 (Negative)')
plt.hist(pos_scores, bins=30, alpha=0.6, color='green', label='Human label = 1 (Positive)')

plt.title("Distribution of VADER Compound Scores by Human Label")
plt.xlabel("VADER Compound Score")
plt.ylabel("Frequency")
plt.legend()
plt.show()


# Create side-by-side plots
plt.figure(figsize=(14,5))

# Plot 1 — Negative class
plt.subplot(1, 2, 1)
plt.hist(neg_scores, bins=30, color='red', alpha=0.7)
plt.title("VADER Compound Distribution\nHuman Label = 0 (Negative)")
plt.xlabel("VADER Compound Score")
plt.ylabel("Frequency")
plt.xlim(-1, 1)

# Plot 2 — Positive class
plt.subplot(1, 2, 2)
plt.hist(pos_scores, bins=30, color='green', alpha=0.7)
plt.title("VADER Compound Distribution\nHuman Label = 1 (Positive)")
plt.xlabel("VADER Compound Score")
plt.ylabel("Frequency")
plt.xlim(-1, 1)

plt.tight_layout()
plt.show()

#%% approach 2: check the classification agreement 
""" 
The widely accepted, research-standard threshold is:
compound > 0.05 → positive
compound < -0.05 → negative
otherwise → neutral

Since human-annotated labels are binary (0 or 1), we will map the VADER scores to the same binary labels:
- VADER score > 0.05 → 1 (Positive)
- VADER score <= 0.05 → 0 (Negative)

Reference: 
Hutto, C.J. & Gilbert, E.E. (2014).  
VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text.  
Proceedings of ICWSM (International Conference on Web and Social Media).

https://ojs.aaai.org/index.php/ICWSM/article/view/14550

""" 

def vader_to_binary(score):
    # Official VADER rule:
    # compound > 0.05 → positive (1)
    # otherwise → negative (0)
    return 1 if score > 0.05 else 0

df["vader_binary"] = df["vader_compound"].apply(vader_to_binary)

print(df['vader_binary'].value_counts(dropna=False)) 
print(pd.crosstab(df['label'], df['vader_binary'], dropna=False))  


# %%
def vader_to_three(score):
    if score > 0.05:
        return "pos"
    elif score < -0.05:
        return "neg"
    else:
        return "neu"

df["vader_3class"] = df["vader_compound"].apply(vader_to_three)

print(df['vader_3class'].value_counts(dropna=False)) 
print(pd.crosstab(df['vader_3class'], df['label'], dropna=False))  
# %%
pearson = df["label"].corr(df["vader_compound"])
print("Pearson correlation:", pearson) 
# %%
