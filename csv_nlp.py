
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer 
import re
import pandas as pd

lemmatizer = WordNetLemmatizer()

file = pd.read_csv("test2_use_review.csv")
reviews = []
rating = list(file['stars'])
filtered_reviews = []
for i in file['text']:
    reviews.append(i.split(" "))
for w in reviews:
    filtered_sentence = []
    lemma_list = []
    for word in w:
        if word not in stop_words:
            filtered_sentence.append(word)
    for j in filtered_sentence:
            j = re.sub(r'[^\w]','',j)
            root_word = lemmatizer.lemmatize(j)
            lemma_list.append(root_word)
    filtered_reviews.append(lemma_list)
ratings = []

for star in rating:
    if float(star) <= 3:
        ratings.append(0)
    if float(star) > 3:
        ratings.append(1)
        
new_review = []

df = pd.DataFrame([filtered_reviews, ratings])
df = df.transpose()

df.to_csv(r'C:/Users/hzamb/Desktop/Miracle Software 2019/sentiment_analysis/nlp_test3_use.csv',index=False)