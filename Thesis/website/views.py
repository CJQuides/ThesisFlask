from flask import Blueprint, render_template, request, flash, jsonify, session, redirect, url_for
from .models import StudentsInfo, Result
from . import db
import json

import pandas as pd
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

views = Blueprint('views', __name__)

#app.secret_key = 'mysecretkey'

@views.route('/', methods=['GET', 'POST'])
def home():
    session['status'] = 'out'
    session['position'] = ''
    return render_template("home.html")

"""  """

@views.route('/addSubs', methods=['GET', 'POST'])
def addSubs():
    session['position'] = 'student'
    stuInfos = StudentsInfo.query.filter_by(student_id=session['userId'])
    
    if request.method == 'POST': 
        for i in range(0, session['numCourse']):
            x=str(i)
            name = request.form.get('name'+x) 
            course = request.form.get('course'+x)
            status = request.form.get('status'+x)
            
            section = request.form.get('section'+x) 
            college = request.form.get('college'+x)
            campus = request.form.get('campus'+x)

            addCourse = StudentsInfo(professorName=name, enrolledCourse=course, evaluationStatus=status, studentSection=section, studentCollege=college, studentCampus=campus, student_id=session['userId'])  #providing the schema for the note 
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
                
            #flash('Note added!', category='success')

            ###

        session['status'] = 'in'
        #session['userId'] = session['student']
        #session['status'] = 'in'
        return redirect(url_for('views.chooseProf'))

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

        rate1 = request.form.get('ARate1')
        rate2 = request.form.get('ARate2')
        rate3 = request.form.get('ARate3')
        rate4 = request.form.get('ARate4')
        rate5 = request.form.get('ARate5')

        rate1 = int(rate1)
        rate2 = int(rate2)
        rate3 = int(rate3)
        rate4 = int(rate4)
        rate5 = int(rate5)

        rateA = (rate1+rate2+rate3+rate4+rate5)/5

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
            if resultExist.catA == None:
                overAllResult = (rateA)

                if overAllResult == 3:
                    sentimentResult = "Neutral"
                elif overAllResult < 3:
                    sentimentResult = "Poor Performance"
                else:
                    sentimentResult = "Excellent Performance"

                resultExist.catA = rateA
                db.session.commit()

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
                resultExist.catA = (resultExist.catA + rateA) / 2
                overAllResult = (resultExist.catA)
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

                if overAllResult == 3:
                    sentimentResult = "Neutral"
                elif overAllResult < 3:
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
                        resultExist.comments = resultExist.comments + " " + saveComment
                        db.session.commit()
                        
               
                resultExist.studentsAnswered = newAnswered
                db.session.commit()

            evalStat = StudentsInfo.query.filter_by(professorName=session['currentFacultyName'], enrolledCourse=session['currentCourse'],student_id=session['userId']).first()            
            evalStat.evaluationStatus = "YES"
            db.session.commit()

            flash('Account created!', category='success')
            return redirect(url_for('views.chooseProf'))

    return render_template("form.html")

@views.route('/showRec')
def showRec():
    records = Result.query.all()

    return render_template("showRec.html",records=records)

url = ("https://raw.githubusercontent.com/CJQuides/MyThesisss/main/Comments-Dataset%20-%20Eng.csv")

dfSen = pd.read_csv(url)
dfSen = pd.DataFrame({'Comments': dfSen['Comments'], 'Result':dfSen['Result']})

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

    review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
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
model=gensim.models.Word2Vec(words,min_count=1,epochs=75) #,window=3,min_count=1,epochs=20

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
        for i in range(0, len(commentsDb)):        
            review = re.sub('[^a-zA-Z]', ' ', commentsDb)
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
            
            vector1 = sum(model.wv[w] for w in tokens1 if w in model.wv.key_to_index)
            vector2 = sum(model.wv[w] for w in tokens2 if w in model.wv.key_to_index)
            
            similarity = nltk.cluster.util.cosine_distance(vector1, vector2)
            similarity = 1 - similarity
        else:
            saveComm = "No"
            break
    
    if similarity > 0.70:
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