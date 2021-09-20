import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
# from sklearn.externals import joblib
import pickle
dataset = pd.read_csv('Reviews.csv')
dataf = dataset[['Summary','Score', 'Text']]
dataf['Sentiment'] = dataf['Score'].apply(lambda rating : +1 if rating > 3 else (0 if rating == 3 else -1))
import nltk
nltk.download('stopwords')
nltk.download('punkt')
# Data Cleaning
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

stopwords = set(stopwords.words('english'))
snowballstemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()
processed_positive_words_list = []
processed_negative_words_list = []
processed_neutral_words_list = []
processed_sentence_list = []

for i in dataf.index:
    this_sentence = []
    stemmedwords = []
    text = dataf.iloc[i]['Text']
    
    # Remove HTML Tags
    text = re.sub('<.*?>', ' ', text) 
    
    # Clear punctuation and numbers
    text = re.sub('[^A-Za-z]+', ' ', text)
    
    # Convert all uppercase characters into lowercase
    text = text.lower()
    
    # Tokenize string
    # Removing stopwords
    # Stemming words
    # Checking wordlength
    for words in word_tokenize(text):
        if len(words) > 1 and words not in stopwords:
            stemmedwords.append(snowballstemmer.stem(words))
            
    if dataf.iloc[i]['Sentiment'] == 1:
        processed_positive_words_list += (stemmedwords)
    elif dataf.iloc[i]['Sentiment'] == -1:                
        processed_negative_words_list += (stemmedwords)
    else:
        processed_neutral_words_list += (stemmedwords)

                
    # Joining words
    clean_sentence = " ".join(stemmedwords)
    processed_sentence_list.append(clean_sentence)

dataf['Cleaned_Text'] = processed_sentence_list

# Splitting dataset into train and test
index = dataf.index
dataf['random_number'] = np.random.randn(len(index))
train = dataf[dataf['random_number'] <= 0.8]
test = dataf[dataf['random_number'] > 0.8]

# Feature Extraction
# Bag of words
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(token_pattern = r'\b\w+\b')
train_matrix = vectorizer.fit_transform(train['Cleaned_Text'])
test_matrix = vectorizer.transform(test['Cleaned_Text'])

import pickle
pickle.dump(vectorizer, open('tranform.pkl', 'wb'))
X_train = train_matrix
X_test = test_matrix
y_train = train['Sentiment']
y_test = test['Sentiment']
# Model Training
# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(multi_class='multinomial')
lr.fit(X_train,y_train)
filename = 'nlp_model.pkl'
pickle.dump(lr, open(filename, 'wb'))

# df= pd.read_csv("cleaned_data.csv", encoding="latin-1")
# #df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
# # Features and Labels
# #df['label'] = df['class'].map({'ham': 0, 'spam': 1})
# df.head()
# X = df['Cleaned_Text']
# y = df['Sentiment']

# # Extract Feature With CountVectorizer
# cv = CountVectorizer()
# X = cv.fit_transform(X) # Fit the Data

# pickle.dump(cv, open('tranform1.pkl', 'wb'))


# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# #Naive Bayes Classifier
# from sklearn.naive_bayes import MultinomialNB

# clf = MultinomialNB()
# clf.fit(X_train,y_train)
# clf.score(X_test,y_test)
# filename = 'nlp_model1.pkl'
# pickle.dump(clf, open(filename, 'wb'))
#lolol
