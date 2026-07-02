from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.preprocessing import OneHotEncoder
import keras.layers
from keras.models import model_from_json
import pickle
import os
from sklearn.preprocessing import normalize

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM

main = Tk()
main.title("DETECTION OF FAKE NEWS THROUGH IMPLEMENTATION OF DATA SCIENCE APPLICATION")
main.geometry("1300x1200")

global filename
global X, Y
global tfidf_X_train, tfidf_X_test, tfidf_y_train, tfidf_y_test
global tfidf_vectorizer
global accuracy,error

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

textdata = []
labels = []
global classifier


def cleanPost(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

def uploadDataset():
    global dataset, textdata, labels
    filename = filedialog.askopenfilename(initialdir="TwitterNewsData")
    text.delete('1.0', END)
    dataset = pd.read_csv(filename, low_memory=False) # ✅ prevents dtype warning
    textdata, labels = [], []
    for i in range(len(dataset)):
        msg = dataset.at[i, 'text'] # ✅ fixed: use [] not ()
        label = dataset.at[i, 'target']
        textdata.append(msg)
        labels.append(label)
    text.insert(END, filename + " loaded\n")
    text.insert(END, "Total records found in dataset : " + str(len(dataset)) + "\n")


def preprocess():
    text.delete('1.0', END)
    global X, Y
    global tfidf_vectorizer
    global tfidf_X_train, tfidf_X_test, tfidf_y_train, tfidf_y_test
    
    stopwords_list = nltk.corpus.stopwords.words("english")
    tfidf_vectorizer = TfidfVectorizer(
        stop_words=stopwords_list, 
        use_idf=True, 
        ngram_range=(1,2),
        smooth_idf=False, 
        norm=None, 
        decode_error='replace', 
        max_features=200
    )
    
    tfidf = tfidf_vectorizer.fit_transform(textdata).toarray()        
    try:
         feature_names = tfidf_vectorizer.get_feature_names_out()
    except AttributeError:  # for older sklearn
         feature_names = tfidf_vectorizer.get_feature_names()

    df = pd.DataFrame(tfidf, columns=feature_names)
    text.insert(END, str(df))
    print("TF-IDF shape:", df.shape)

    df = df.values
    X = normalize(df)
    Y = np.asarray(labels)

    le = LabelEncoder()
    Y = le.fit_transform(Y)
    Y = Y.reshape(-1, 1)

    # Reshape X into 3D for LSTM: (samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    tfidf_X_train, tfidf_X_test, tfidf_y_train, tfidf_y_test = train_test_split(X, Y, test_size=0.2)

    # Ensure both train and test are also 3D
    tfidf_X_train = tfidf_X_train.reshape((tfidf_X_train.shape[0], tfidf_X_train.shape[1], 1))
    tfidf_X_test  = tfidf_X_test.reshape((tfidf_X_test.shape[0], tfidf_X_test.shape[1], 1))

    text.insert(END,"\n\nTotal News found in dataset : "+str(len(X))+"\n")
    text.insert(END,"Total records used to train machine learning algorithms : "+str(len(tfidf_X_train))+"\n")
    text.insert(END,"Total records used to test machine learning algorithms  : "+str(len(tfidf_X_test))+"\n")


def runLSTM():
    text.delete('1.0', END)
    global classifier
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
        classifier.load_weights("model/model_weights.weights.h5")
        print(classifier.summary())

        f = open('model/history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy'][9] * 100
        text.insert(END,"LSTM Fake News Detection Accuracy : "+str(acc)+"\n\n")
        text.insert(END,'LSTM Model Summary can be seen in black console for layer details\n')

    else:
        lstm_model = Sequential()
        lstm_model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
        lstm_model.add(Dropout(0.2))

        lstm_model.add(LSTM(128))
        lstm_model.add(Dropout(0.2))

        lstm_model.add(Dense(32, activation='relu'))
        lstm_model.add(Dropout(0.2))

        lstm_model.add(Dense(2, activation='softmax'))
        
        lstm_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        hist = lstm_model.fit(tfidf_X_train, tfidf_y_train, epochs=10, validation_data=(tfidf_X_test, tfidf_y_test))

        classifier = lstm_model
        classifier.save_weights('model/model_weights.weights.h5')
        model_json = classifier.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)

        accuracy = hist.history
        f = open('model/history.pckl', 'wb')
        pickle.dump(accuracy, f)
        f.close()

        acc = accuracy['accuracy'][9] * 100
        text.insert(END,"LSTM Accuracy : "+str(acc)+"\n\n")
        text.insert(END,'LSTM Model Summary can be seen in black console for layer details\n')
        print(lstm_model.summary())
        

    
def graph():
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    acc = data['accuracy']
    loss = data['loss']
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Epcchs')
    plt.ylabel('Accuracy/Loss')
    plt.plot(acc, 'ro-', color = 'green')
    plt.plot(loss, 'ro-', color = 'blue')
    plt.legend(['Accuracy','Loss'], loc='upper left')
    #plt.xticks(wordloss.index)
    plt.title('LSTM Model Accuracy & Loss Graph')
    plt.show()

def predict():
    print(">>> Running NEW predict() function <<<")
    testfile = filedialog.askopenfilename(initialdir="TwitterNewsData",
                                          filetypes=[("Text/CSV files", "*.txt *.csv")])
    text.delete('1.0', END)

    testData = []

    if testfile.endswith(".csv"):
        df = pd.read_csv(testfile)
        if "text" not in df.columns:
            text.insert(END, "CSV file must have a 'text' column!\n")
            return
        testData = df["text"].dropna().astype(str).tolist()
    elif testfile.endswith(".txt"):
        with open(testfile, "r", encoding="utf-8") as f:
           lines = f.readlines()   # read all lines
        testData = []
        for line in lines:
           s = str(line).strip()   # force to string
        if s != "":
            testData.append(s)

    
    else:
        text.insert(END, "Unsupported file type. Please select .csv or .txt\n")
        return

    for msg1 in testData:
        msg1 = str(msg1)
        print(msg1)

        review = cleanPost(msg1.lower().strip())
        if review.strip() == "":
            text.insert(END, msg1 + " === Skipped (empty after preprocessing)\n\n")
            continue

        testReview = tfidf_vectorizer.transform([review]).toarray()
        testReview = testReview.reshape((1, testReview.shape[1], 1))

        pred_probs = classifier.predict(testReview)
        pred_class = int(np.argmax(pred_probs, axis=1)[0])  # ✅ always scalar int

        print(pred_probs, "-> class:", pred_class)

        if pred_class == 0:   # ✅ safe
            text.insert(END, msg1 + " === Given news predicted as GENUINE\n\n")
        else:
            text.insert(END, msg1 + " === Given news predicted as FAKE\n\n")


    
font = ('times', 15, 'bold')
title = Label(main, text='DETECTION OF FAKE NEWS THROUGH IMPLEMENTATION OF DATA SCIENCE APPLICATION')
title.config(bg='gold2', fg='thistle1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload Fake News Dataset", command=uploadDataset)
uploadButton.place(x=20,y=100)
uploadButton.config(font=ff)


processButton = Button(main, text="Preprocess Dataset", command=preprocess)
processButton.place(x=20,y=150)
processButton.config(font=ff)

dtButton = Button(main, text="Run LSTM Algorithm", command=runLSTM)
dtButton.place(x=20,y=200)
dtButton.config(font=ff)

graphButton = Button(main, text="Accuracy & Loss Graph", command=graph)
graphButton.place(x=20,y=250)
graphButton.config(font=ff)

predictButton = Button(main, text="Test News Detection", command=predict)
predictButton.place(x=20,y=300)
predictButton.config(font=ff)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=330,y=100)
text.config(font=font1)

main.config(bg='DarkSlateGray1')
main.mainloop()
