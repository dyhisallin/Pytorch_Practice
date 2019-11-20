#%% md
#### load data with ngrams

#%%
import torch
import torchtext
from torchtext.datasets import text_classification
NGRAMS = 2
import os
import sys
project_dir = sys.path[1].replace('\\', '/')

#%%
BATCH_SIZE = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset, test_dataset = text_classification.DATASETS['AmazonReviewPolarity'](root=os.path.join(project_dir, 'data/'), ngrams=NGRAMS, vocab=None)


