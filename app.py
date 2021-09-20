from flask import Flask,request, url_for, redirect, render_template
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
# import numpy as np

app = Flask(__name__)

filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('tranform.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("main.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == 'POST':

        text = request.form['text1']
        
        stopwords1 = set(stopwords.words('english'))
        snowballstemmer = SnowballStemmer('english')
        lemmatizer = WordNetLemmatizer()


      # for i in dataf.index:
        this_sentence = []
        stemmedwords = []
        #text = dataf.iloc[i]['Text']
        
        # Remove HTML Tags
        text = re.sub('<.*?>', ' ', text) 
        
        # Clear punctuation and numbers
        text = re.sub('[^A-Za-z]+', ' ', text)
        
        # Convert all uppercase characters into lowercase
        text = text.lower()
        
       
        for words in word_tokenize(text):
            if len(words) > 1 and words not in stopwords1:
                stemmedwords.append(snowballstemmer.stem(words))  
        # Joining words
        clean_sentence = " ".join(stemmedwords)
        #processed_sentence_list.append(clean_sentence)
        data=[clean_sentence]
        vect = cv.transform(data).toarray()
        my_pred= clf.predict(vect)
    
    return render_template('results.html',prediction = my_pred)


if __name__ == '__main__':
    app.run(debug=True)
