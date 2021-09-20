# REVIEWS SENTIMENT ANALYSIS 
 (SCREENSHOTS AND FLOWCHART INCLUDED)
A machine learning end to end flask web app for sentiment analysis model created using Scikit-learn,flask.
The project uses libraries like : <br />
Flask <br />
Sklearn <br />
Requests <br />
NLTK <br />
RE <br />
### LET'S TALK ABOUT SENTIMENT ANALYSIS <br />
Sentiment analysis, an important area in Natural Language Processing, is the process of automatically detecting affective states of text. Sentiment analysis is widely applied to voice-of-customer materials such as product reviews in online shopping websites like Amazon, movie reviews or social media. It can be just a basic task of classifying the polarity of a text as being positive/negative or it can go beyond polarity, looking at sentiment states etc. <br />
Sentiment analysis refers to analyzing an opinion or feelings about something using data like text or images, regarding almost anything. Sentiment analysis helps companies in their decision-making process. For instance, if public sentiment towards a product is not so good, a company may try to modify the product or stop the production altogether in order to avoid any losses. <br />
There are many sources of public sentiment e.g. public interviews, opinion polls, surveys, etc. However, with more and more people joining social media platforms, websites like Facebook and Twitter can be parsed for public sentiment. <br />
<br />

this project aims to analyze and categorize the reviews of the customers into 3 categories i.e. POSITIVE,NEUTRAL,NEGATIVE. i have trained the raw dataset (https://www.kaggle.com/snap/amazon-fine-food-reviews) from scratch which includes data pre processing, feature selection , data cleaning using various methods such as : stop words removal, lemmatization,stemming, removal of punctuation marks, full stops, hastags etc . The data is ssplit into training and testing based on some random numbers.After the datacleaning process i have used BAG OF WORDS to handle the text.after standard scaling I have used LOGISTIC REGRESSION to train and fit the model.all the code is saved in the code.py file.

The deployment is done using FLASK,HEROKU.

SCREENSHOTS OF THE WEB APP <br/> 
### entering review:<br />
![image](https://user-images.githubusercontent.com/68751708/134017239-cc868ef4-cb03-4dbd-97a7-1d416f768ef9.png)<br/>
### result for POSITIVE REVIEW:<br />
![image](https://user-images.githubusercontent.com/68751708/134017282-1e5f4e2f-b0fc-47cf-9bd5-d4fe10f90dcb.png)
### result for NEGATIVE REVIEW:<br />
![image](https://user-images.githubusercontent.com/68751708/134017885-e265d68b-dfea-4fc5-ad87-6f6fc860e7d1.png)
### result for NEUTRAL REVIEW:<br />
![image](https://user-images.githubusercontent.com/68751708/134017921-99535c47-39f4-4f88-a9dd-22fb5b814e2c.png)<br/>
### methodology<br/>
![image](https://user-images.githubusercontent.com/68751708/134021289-35bcc4b8-5011-4c4d-8859-524106722180.png)

