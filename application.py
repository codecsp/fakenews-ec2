from flask import Flask, abort, jsonify, request, render_template
import joblib
import sklearn
from feature import *
import json

pipeline = joblib.load('./pipeline.sav')

application = Flask(__name__)

@application.route('/')
def home():
    return render_template('index.html')

@application.route('/api',methods=['POST'])
def get_delay():

    result=request.form
    query_title = result['title']
    query_author = result['author']
    query_text = result['maintext']
    print(query_text)
    query = get_all_query(query_title, query_author, query_text)
    user_input = {'query':query}
    pred = pipeline.predict(query)
    print(pred)
    #pred=[0]
    dic = {1:'Real',0:'Fake'}
    return render_template('show.html',inf=dic[pred[0]])
    #return f'<html><body><h1>{dic[pred[0]]}</h1> <form action="/"> <button type="submit">back </button> </form></body></html>'


if __name__ == '__main__':
    application.run(host='0.0.0.0', port=8080, debug=True)
