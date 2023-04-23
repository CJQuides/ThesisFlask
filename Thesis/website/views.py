from flask import Blueprint, render_template, request, flash, jsonify, session, redirect, url_for
from .models import StudentsInfo, Result
from . import db
import json

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

import gensim
from nltk import sent_tokenize
from gensim.utils import simple_preprocess
from nltk.tokenize import word_tokenize

import pickle
import requests

views = Blueprint('views', __name__)

#app.secret_key = 'mysecretkey'

"""  """

@views.route('/addSubs', methods=['GET', 'POST'])
def addSubs():
    session['position'] = 'admin'
    stuInfos = StudentsInfo.query.filter_by(student_id=session['userId'])
    
    if request.method == 'POST': 
        for i in range(0, session['numCourse']):
            x=str(i)
            name = request.form.get('name'+x) 
            course = request.form.get('course'+x)
            
            section = request.form.get('section'+x) 
            college = request.form.get('college'+x)
            campus = request.form.get('campus'+x)

            addCourse = StudentsInfo(professorName=name, enrolledCourse=course, evaluationStatus="NO", studentSection=section, studentCollege=college, studentCampus=campus, student_id=session['userId'])  #providing the schema for the note 
            db.session.add(addCourse) 
            db.session.commit() 
                    
            resultExists = Result.query.filter_by(facultyName=name, course=course, facultyClass=section, college=college, campus=campus, schoolYear="2022-2023", semester="Second").first()

            if resultExists:
                x=resultExists.studentsAnswered.split()

                y=x[0]

                z=x[2]
                z=int(z)
                z=z+1
                z=str(z)

                newAnswered= y+" / "+z                
                resultExists.studentsAnswered = newAnswered
                db.session.commit()
            else:
                addCourse = Result(facultyName=name, course=course, facultyClass=section, college=college, campus=campus, schoolYear="2022-2023", semester="Second", studentsAnswered="0 / 1")  #providing the schema for the note 
                db.session.add(addCourse) 
                db.session.commit()            

            ###

        session['status'] = 'in'
        #session['userId'] = session['student']
        #session['status'] = 'in'
        #flash('Account Created!', category='success')
        return redirect(url_for('auth.studentSignUp'))

    return render_template("addSubs.html",userNow=stuInfos)


@views.route('/chooseProf', methods=['GET', 'POST'])
def chooseProf():
    session['numCourse'] = 0
    stuInfos = StudentsInfo.query.filter_by(student_id=session['userId'])

    #session.pop('currentFacultyName', None)
    #session.pop('course', None)

    if request.method == 'POST': 
        facultyName = request.form.get('facultyNames')
        course = request.form.get('course')
        campus = request.form.get('campus')
        college = request.form.get('college')
        section = request.form.get('class')
        evalStatus = request.form.get('evalStatus')

        session['currentFacultyName'] = facultyName
        session['currentCourse'] = course
        session['currentCampus'] = campus
        session['currentCollege'] = college
        session['currentClass'] = section
        session['evalStatus'] = evalStatus
        return redirect(url_for('views.form'))

    return render_template("chooseProf.html", stuInfo=stuInfos)


@views.route('/form', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        comment = request.form.get('comment')

        rateA1 = request.form.get('ARate1')
        rateA2 = request.form.get('ARate2')
        rateA3 = request.form.get('ARate3')
        rateA4 = request.form.get('ARate4')
        rateA5 = request.form.get('ARate5')

        rateA1, rateA2, rateA3, rateA4, rateA5 = [int(x) for x in (rateA1, rateA2, rateA3, rateA4, rateA5)]

        rateA = (rateA1+rateA2+rateA3+rateA4+rateA5)/5

        """ A """

        rateB1 = request.form.get('BRate1')
        rateB2 = request.form.get('BRate2')
        rateB3 = request.form.get('BRate3')
        rateB4 = request.form.get('BRate4')
        rateB5 = request.form.get('BRate5')

        rateB1, rateB2, rateB3, rateB4, rateB5 = [int(x) for x in (rateB1, rateB2, rateB3, rateB4, rateB5)]

        rateB = (rateB1+rateB2+rateB3+rateB4+rateB5)/5

        """ B """
        
        rateC1 = request.form.get('CRate1')
        rateC2 = request.form.get('CRate2')
        rateC3 = request.form.get('CRate3')
        rateC4 = request.form.get('CRate4')
        rateC5 = request.form.get('CRate5')

        rateC1, rateC2, rateC3, rateC4, rateC5 = [int(x) for x in (rateC1, rateC2, rateC3, rateC4, rateC5)]

        rateC = (rateC1+rateC2+rateC3+rateC4+rateC5)/5

        """ C """

        rateD1 = request.form.get('DRate1')
        rateD2 = request.form.get('DRate2')
        rateD3 = request.form.get('DRate3')
        rateD4 = request.form.get('DRate4')
        rateD5 = request.form.get('DRate5')
        rateD6 = request.form.get('DRate6')

        rateD1, rateD2, rateD3, rateD4, rateD5, rateD6 = [int(x) for x in (rateD1, rateD2, rateD3, rateD4, rateD5, rateD6)]

        rateD = (rateD1+rateD2+rateD3+rateD4+rateD5+rateD6)/6

        """ D """

        rateE1 = request.form.get('ERate1')

        rateE = int(rateE1)

        """ Over All """

        rateAPS = (rateA+rateB+rateC+rateD+rateE) / 5

        rateA, rateB, rateC, rateD, rateAPS = [round(x, 2) for x in (rateA, rateB, rateC, rateD, rateAPS)]

        #sentimentValue, saveComment = SentenceAnalysis(comment)   

        """ sentimentValue = 3

        saveComment = "No Algo" """

        resultExist = Result.query.filter_by(facultyName=session['currentFacultyName'],course=session['currentCourse'], facultyClass=session['currentClass'], campus=session['currentCampus'], college=session['currentCollege'], schoolYear="2022-2023", semester="Second").first()#,facultyClass

        dbComments = resultExist.comments

        sentimentValue, saveComment = SentenceAnalysis(comment, dbComments)  
        
        #Update Record
        if resultExist:
            x=resultExist.studentsAnswered.split()

            y=x[0]
            y=int(y)
            y=y+1
            y=str(y)

            z=x[2]

            newAnswered= y+" / "+z 

            #if no one answered yet
            if resultExist.APS == None:

                resultExist.catA = rateA
                db.session.commit()

                resultExist.catB = rateB
                db.session.commit()

                resultExist.catC = rateC
                db.session.commit()

                resultExist.catD = rateD
                db.session.commit()

                resultExist.APS = rateAPS
                db.session.commit()

                logResult = predictResult(rateA, rateB, rateC, rateD, rateAPS)

                if logResult == 1:
                    sentimentResult = "Poor Performance"
                else:
                    sentimentResult = "Excellent Performance"

                resultExist.result = sentimentResult
                db.session.commit()

                resultExist.studentsAnswered = newAnswered
                db.session.commit()

                resultExist.sentimentTracker = sentimentValue
                db.session.commit()
                
                if(sentimentValue == 3):
                    resultExist.senAnalysis = "Neutral"
                    db.session.commit
                elif(sentimentValue > 3):
                    resultExist.senAnalysis = "Positive"
                    db.session.commit()
                else:
                    resultExist.senAnalysis = "Negative"
                    db.session.commit
                
                
                if len(saveComment) > 3:
                    resultExist.comments = saveComment
                    db.session.commit()
                    
            else:
                # Update 
                resultExist.catA = round(((resultExist.catA + rateA) / 2), 2)
                db.session.commit()
                
                resultExist.catB = round(((resultExist.catB + rateB) / 2), 2)
                db.session.commit()
                
                resultExist.catC = round(((resultExist.catC + rateC) / 2), 2)
                db.session.commit()
                
                resultExist.catD = round(((resultExist.catD + rateD) / 2), 2)
                db.session.commit()
                
                resultExist.APS = round(((resultExist.APS + rateAPS) / 2), 2)
                db.session.commit()

                newTracker = (resultExist.sentimentTracker + sentimentValue) / 2
                resultExist.sentimentTracker = newTracker
                db.session.commit()
                
                if(newTracker == 3):
                    resultExist.senAnalysis = "Neutral"
                    db.session.commit
                elif(newTracker > 3):
                    resultExist.senAnalysis = "Positive"
                    db.session.commit()
                else:
                    resultExist.senAnalysis = "Negative"
                    db.session.commit

                newLogResult = predictResult(resultExist.catA, resultExist.catB, resultExist.catC, resultExist.catD, resultExist.APS)

                if newLogResult == 1:
                    sentimentResult = "Poor Performance"
                else:
                    sentimentResult = "Excellent Performance"

                resultExist.result = sentimentResult
                db.session.commit()

                if len(saveComment) > 3:
                    if resultExist.comments == None:
                        resultExist.comments = saveComment
                        db.session.commit()
                    else:
                        last_sentence = dbComments[-1]  # Get the last sentence in the list
                        punctuation = last_sentence[-1]  

                        if punctuation == ".":
                            resultExist.comments = resultExist.comments + " " + saveComment
                            db.session.commit()
                        else:
                            resultExist.comments = resultExist.comments + ". " + saveComment
                            db.session.commit()
                        
                resultExist.studentsAnswered = newAnswered
                db.session.commit()

            evalStat = StudentsInfo.query.filter_by(professorName=session['currentFacultyName'], enrolledCourse=session['currentCourse'],student_id=session['userId']).first()            
            evalStat.evaluationStatus = "YES"
            db.session.commit()

            #flash('Evaluation Submitted!', category='success')
            return redirect(url_for('views.chooseProf'))

    return render_template("form.html")

@views.route('/showRec')
def showRec():
    records = Result.query.all()

    return render_template("showRec.html",records=records)

urlCom = ("https://raw.githubusercontent.com/CJQuides/MyThesisss/main/Comments-Dataset.csv")
urlTagStop = ("https://raw.githubusercontent.com/CJQuides/MyThesisss/main/tagalog_stopwords.txt")

tlStopwords = []

response = requests.get(urlTagStop)
content = response.text

tlStopwords = content.split('\r\n')

allStopWords = set(stopwords.words('english') + tlStopwords)
    
dfSen = pd.read_csv(urlCom, encoding='cp1252')
dfSen = pd.DataFrame({'Comments': dfSen['Comments'], 'Result':dfSen['Result']})

urlWord2Vec = "https://raw.githubusercontent.com/CJQuides/MyThesisss/main/word2vec_model"
response = requests.get(urlWord2Vec)
data = response.content

summaryPickle = pickle.loads(data)

#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('omw-1.4')

#Lemmatize
corpus = []
lemmatizer = WordNetLemmatizer()
for i in range(0, len(dfSen)):
    review = re.sub('[^a-zA-Z]', ' ', dfSen['Comments'][i])
    review = review.lower()
    review = review.split()

    review = [lemmatizer.lemmatize(word) for word in review if not word in allStopWords or word in ['not', 'can']]
    review = ' '.join(review)
    corpus.append(review)

## tokenize
allWords=[]
for i in corpus:
    words=nltk.word_tokenize(i)
    for word in words:
        allWords.append(word)

words=[]
for sent in corpus:
    sent_token=sent_tokenize(sent)
    for sent in sent_token:
        words.append(simple_preprocess(sent))

### Lets train Word2vec from scratch
#model=gensim.models.Word2Vec(words,min_count=1,epochs=75) #,window=3,min_count=1,epochs=20

#Creating the Bag of Words model
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()
y=dfSen['Result']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)

MNB_model = MultinomialNB().fit(X_train, y_train)

def SentenceAnalysis(userInput, commentsDb):

    # Preprocess Comments in the database
    dbComCorpus = []    
    saveComm = ""

    if commentsDb != None:
        commentsDb=sent_tokenize(commentsDb)
        for i in range(0, len(commentsDb)):        
            review = re.sub('[^a-zA-Z]', ' ', commentsDb[i])
            review = review.lower()
            review = review.split()

            review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
            review = ' '.join(review)
            dbComCorpus.append(review)
    else:
        saveComm = "Yes"

    #User Input
    userComment = userInput

    userCorpus = []

    userCommentLow = userComment
    userCommentLow = userCommentLow.lower()

    #Lemmatizing the user's input

    sentimentValue = 0
    if userCommentLow == "none" or userCommentLow == "none." or userCommentLow == "n/a" or userCommentLow == "n/a.":
        sentimentValue = 3
    else:
        review = re.sub('[^a-zA-Z]', ' ', userComment)
        review = review.lower()
        review = review.split()

        review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        userCorpus.append(review)

    similarity = 0
    for i in dbComCorpus:
        if len(userCorpus) > 0:
            tokens2 = [w.lower() for w in word_tokenize(userCorpus[0])]
            tokens1 = [w.lower() for w in word_tokenize(i)]
            
            vector1 = sum(summaryPickle.wv[w] for w in tokens1 if w in summaryPickle.wv.key_to_index)
            vector2 = sum(summaryPickle.wv[w] for w in tokens2 if w in summaryPickle.wv.key_to_index)
            
            similarity = nltk.cluster.util.cosine_distance(vector1, vector2)
            similarity = 1 - similarity
            
            if np.all(np.isnan(similarity)):
                saveComm = "No"
                sentimentValue = 3
            else:
                if similarity > 0.75:
                    saveComm = "No"
                    break
                else:
                    saveComm = "Yes"
        else:
            saveComm = "No"
            break
            
    if np.all(np.isnan(similarity)):
        saveComm = "No"
        sentimentValue = 3
    else:
        if similarity > 0.75:
            saveComm = "No"
        else:
            saveComm = "Yes"

    #tokenize user sentence
    userWords=[]
    for i in userCorpus:
        words=nltk.word_tokenize(i)
        for word in words:
            userWords.append(word)
            
    #check if user words exists in model
    newWords = []
    seen = []

    for row in userWords:
        if row not in allWords:
            newWords.append(row)
        else:
            seen.append(row)
            
    saveSenToDb = ""
    #predicting the result
    if sentimentValue == 0:
        if len(newWords) < 1:
            corpus.append(userCorpus[0])

            X = cv.fit_transform(corpus).toarray()

            userX = X[len(corpus)-1]

            newX = []
            newX.append(userX)

            y_pred=MNB_model.predict(newX)
            predResult = y_pred[0]

            #assigning value
            if predResult == 1:
                sentimentValue = 5
            elif predResult == 0:
                sentimentValue = 1

            #sentimentValue = int(sentimentValue)

            #cleaning dataframe
            dfSave = pd.DataFrame({'Comments': dfSen['Comments'], 'Result':dfSen['Result']})

            #saving the new comment to dataset
            dfSave.loc[len(dfSave)] = [userComment, predResult]

            dfSave.to_csv('Comments-Dataset - Eng.csv')

            if saveComm == "Yes":
                saveSenToDb = userComment
        else:
            if saveComm == "Yes":
                saveSenToDb = userComment

            print("word doesn't exist")
            sentimentValue = 3

            #pickled_model = pickle.load(open('word2vec-google-news-300_model_pickle', 'rb'))

    return sentimentValue, saveSenToDb

urlLogModel = "https://raw.githubusercontent.com/CJQuides/MyThesisss/main/logModel_pickle"
response = requests.get(urlLogModel)
data = response.content

logPickle = pickle.loads(data)

def predictResult(a, b, c, d, aps):
    predMe = np.array([[a],[b],[c],[d],[aps]])
    predMe = predMe.reshape(-1, 5)

    logPred = logPickle.predict(predMe)
    return logPred[0]