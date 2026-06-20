


#%% 
import pandas as pd
import os
print(os.getcwd())

script_dir = os.path.dirname(os.path.abspath(__file__))
print(script_dir)

#%% 
""" we will use a famous dataset (SST)

Stanford Sentiment Treebank (SST)  
https://nlp.stanford.edu/sentiment/

This includes:
5-class labels
phrase-level annotations
parse trees


SST-2 is derived from this dataset by:
removing neutral examples
collapsing 5 classes → 2 classes
keeping only full sentences
"""

#%% 
from datasets import load_dataset
dataset = load_dataset("stanfordnlp/sst2")

#%% 
print(dataset) 

# %%
train = dataset["train"]
print(train)
print(train[0:2]) 
#%% 
print(train[0]["label"])
# %%
for i in range(5):
    print(train[i]["sentence"], train[i]["label"])
# %%
"""
RoBERTa is a pre-trained language model that can be fine-tuned for various NLP tasks, including sentiment analysis.

reads text. understands it with remarkable depth and context. 
then uses that undrestanding to perform tasks like classification, sentiment analysis and more. 

RoBERTa is build on the Transformer architecture. which allows it to:
- understand meaning
- detect sentiment  
- Classify text into categories (Text Classification. Spam detection)
- Extract features for downstream ML models
- Understand context better than older models like LSTMs or bag-of-words
- handle long, complex sentences with nuance 

BERT (Bidirectional Encoder Representations from Transformers) 
- is a model that reads text in both directions at the same time, so it understands meaning with much deeper context than older models.

Variations: 
1. RoBERTa: Robustly Optimized BERT 
2. DistilBERT: A smaller, faster version of BERT 
3. ALBERT: a lite BERT
4. BERT-Large / BERT-Base 
5. Domain-Specific BERT Variants: BioBERT, LegalBERT, ClinicalBERT, SciBERT, Multilingual BERT, TinyBERT/MiniLM  
- 
"""

#%%
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"

classifier = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)

# %%
print(classifier("I love this movie!"))

# %%
for i in range(5):
    text = dataset["validation"][i]["sentence"]
    print(text, classifier(text))


# %%
correct = 0
total = 0

for item in dataset["validation"]:
    text = item["sentence"]
    true_label = item["label"]

    pred = classifier(text)[0]["label"]
    pred_label = 1 if pred == "positive" else 0

    correct += (pred_label == true_label)
    total += 1

print("Accuracy:", correct / total)
# %%
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# collect all true/predicted labels
all_true = []
all_pred = []

for item in dataset["validation"]:
    text = item["sentence"]
    true_label = item["label"]

    pred = classifier(text)[0]["label"]
    pred_label = 1 if pred == "positive" else 0

    all_true.append(true_label)
    all_pred.append(pred_label)

#%% confusion matrix
cm = confusion_matrix(all_true, all_pred, labels=[0, 1])
print("Confusion matrix:")
print(cm)

# cross-tab view 
print("\nCross-tab:")
df = pd.DataFrame({
    "true": all_true,
    "pred": all_pred
})

print(pd.crosstab(
    df["true"],
    df["pred"],
    rownames=["Actual"],
    colnames=["Predicted"],
    dropna=False
))
# %%
errors = []

for item in dataset["validation"]:
    text = item["sentence"]
    true_label = item["label"]
    pred = classifier(text)[0]["label"]
    pred_label = 1 if pred == "positive" else 0

    if pred_label != true_label:
        errors.append((text, true_label, pred_label))

errors[:10]

# %%
"""in the above example, we directly apply pre-trained RoBERTa the validation dataset.

Now, fine-tune RoBERTa on the training dataset and then evaluate it on the validation datset. 

two different model names ---  
r-roberta-base-sentiment-latest → a pretrained sentiment model (analogy: a finished cake)

roberta-base → a general RoBERTa model you fine-tune yourself (analogy: raw incredients (flour, eggs, sugar)) 


it turns out training/refine-tuning a model is very slow, requires gpu. 
so for most analysis, we will use the pretrained model directly. 
"""
#1. load dataset
from datasets import load_dataset
dataset = load_dataset("stanfordnlp/sst2")

#2. load tokennizer 
from transformers import AutoTokenizer
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

#3. tokenize the dataset 
def tokenize(batch):
    return tokenizer(batch["sentence"], padding="max_length", truncation=True)

tokenized = dataset.map(tokenize, batched=True)
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch")

#3b. need to sample down to shorten training time 

import random
# Make sampling reproducible
random.seed(42)
# Total number of training samples
n = len(tokenized["train"])
# Randomly pick 2000 indices
indices = random.sample(range(n), 2000)
# Create the subset
train2000 = tokenized["train"].select(indices)



#%% 4. load RoBERTa for classification 
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

#5. training setup     
from transformers import TrainingArguments, Trainer

# -- this is for GPU. too heavy for CPU
"""  
args = TrainingArguments(
    output_dir="roberta-sst2",
    eval_strategy="epoch", 
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
) """

# -- this is for CPU 
args = TrainingArguments(
    output_dir="outputs",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,

    eval_strategy="epoch",
    save_strategy="no",
    load_best_model_at_end=False,

    logging_steps=100,
    dataloader_num_workers=0,

    fp16=False,
    bf16=False,
)



 
#%% 6. trainer 

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train2000, 
    eval_dataset=tokenized["validation"],
)

#7. train
trainer.train()

#takes 12hours to train the model and another 20 mins to evaluate the model.
""" 
Epoch	Training Loss	Validation Loss
1	0.379484	0.307219
""" 

#%% 8. evaluate 
my_metrics = trainer.evaluate()
print("My fine-tuned model:", my_metrics)

#%% 
"""
we are evaluating again on the validation set. 

but its not redundant step.
earlier, evaluation was done during training. 
now it happens after training is fully complete. serves different purposes.
- it confirms the final model's performance.
- gives metrics in a variable
- ensures reproducibility of the results 

"""

#%% get confusion matrix
pred = trainer.predict(tokenized["validation"])

import numpy as np
y_true = pred.label_ids
y_pred = np.argmax(pred.predictions, axis=1)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
cm

"""
array([[370,  58],
       [ 27, 417]])
this does look better than using pretrained/without fine-tuning. 
"""

# %%
"""
we can also use the Hugging Face pre-trained model as trainer instead of base model.


hf_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"

hf_model = AutoModelForSequenceClassification.from_pretrained(
    hf_model_name,
    num_labels=3   # negative, neutral, positive
)

hf_trainer = Trainer(
    model=hf_model,
    args=args,
    eval_dataset=tokenized["validation"],
    compute_metrics=compute_metrics
)

hf_metrics = hf_trainer.evaluate()

# note, sst2 has binary label. 
# but hf model has 3 labels. 
# we need to map the 3 labels to the 2 labels of sst2 during evaluation step.
 
"""   
