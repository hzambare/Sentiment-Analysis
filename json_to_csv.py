

import pandas as pd

#Shrink JSON file
iterations = 375
file = open('C:/Users/hzamb/Desktop/Miracle Software 2019/sentiment_analysis/use_review.json', 'w+')
for i in range(350, iterations): #662
    with open("review.json") as myfile:
        line = [next(myfile) for x in range(i*7849,(i+1)*7849)]    
        for i in line:
            file.write(i)

reviews_filepath = 'C:/Users/hzamb/Desktop/Miracle Software 2019/sentiment_analysis/use_review.json'

fp = open(reviews_filepath, encoding="ISO-8859-1")
lst = []
reader = pd.read_json(fp, lines=True, chunksize=10000)
for chunk in reader:
    x = pd.DataFrame(chunk)
    lst.append(x)

df_reviews = pd.DataFrame(columns=x.columns)
for item in lst:
    df_reviews = pd.concat([df_reviews, item])

df = df_reviews.drop(['business_id', 'cool', 'date', 'funny', 'review_id', 'useful', 'user_id'], axis=1)

df.to_csv(r'C:/Users/hzamb/Desktop/Miracle Software 2019/sentiment_analysis/test2_use_review.csv',index=False)