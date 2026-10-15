from flask import Flask, render_template, request
import pandas as pd
import sklearn
import itertools
import numpy as np
import seaborn as sb
import re
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib import pyplot as plt
from sklearn.linear_model import PassiveAggressiveClassifier
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
except:
    pass

app = Flask(__name__, template_folder='.', static_folder='./static')

loaded_model = pickle.load(open("model.pkl", 'rb'))
tfidf_v = pickle.load(open("vector.pkl", 'rb'))

lemmatizer = WordNetLemmatizer()
stpwrds = set(stopwords.words('english'))

def fake_news_det(news):
    corpus = []
    review = news
    review = re.sub(r'[^a-zA-Z\s]', '', review)
    review = review.lower()
    review = nltk.word_tokenize(review)
    for y in review:
        if y not in stpwrds:
            corpus.append(lemmatizer.lemmatize(y))
            
    input_data = [' '.join(corpus)]
    vectorized_input_data = tfidf_v.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    
    # Tính % độ tin cậy
    decision_score = loaded_model.decision_function(vectorized_input_data)
    confidence = 1 / (1 + np.exp(-np.abs(decision_score)))
    confidence_percent = round(np.mean(confidence) * 100, 2)
     
    return prediction, confidence_percent

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        message = request.form['news']
        pred, confidence = fake_news_det(message)
        print(f"Mã dự đoán: {pred} | Độ tự tin: {confidence}%")

        if pred[0] == 1:
           result = f"Prediction of the News : Looking Fake⚠ News📰 (Độ tin cậy: {confidence}%)"
        else:
           result = f"Prediction of the News : Looking Real News📰 (Độ tin cậy: {confidence}%)"

        # Gửi trọn vẹn cả prediction và news_text sang trang web
        return render_template("prediction.html", prediction=result, news_text=message)
    else:
        return render_template('prediction.html', prediction="Hãy dán tiêu đề/nội dung tin tức vào ô bên dưới và bấm Predict để kiểm tra!", news_text="")

if __name__ == '__main__':
    # Chạy ở cổng 8080 để tránh lỗi bộ nhớ đệm (cache) của Windows
    app.run(debug=True, port=8080)