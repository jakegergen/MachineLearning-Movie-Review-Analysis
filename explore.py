import numpy as numpy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train =pd.read_csv("train.tsv",sep='\t')
sentiments = train['Sentiment']

sns.countplot(sentiments)
plt.show()
