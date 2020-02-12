from flask import Flask, render_template, send_file
from flaskexample import app
from flask import request
import numpy as np
import pandas as pd
from pandas import DataFrame
import nltk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from xgboost import XGBClassifier
from math import sin, cos, sqrt, atan2
import folium
from folium import plugins

## ====== functions ======
## Machine Learning Models
def ml(lat,lon,df,para): 
  training_data, testing_data, Y_train, Y_test = train_test_split(df, para, test_size=0.33)
  X_train=np.stack((training_data['latitude'].values, training_data['longitude'].values)).transpose()
  X_test=np.stack((testing_data['latitude'].values, testing_data['longitude'].values)).transpose()
#  model = LogisticRegression()
#  model = RandomForestClassifier()
  model = XGBClassifier()
  result = model.fit(X_train, Y_train)
  Y_pred = model.predict(X_test)
  acc = '{:02.4f}'.format(metrics.accuracy_score(Y_test, Y_pred))
  p = '{:02.4f}'.format(metrics.precision_score(Y_test, Y_pred, average='macro'))
  r = '{:02.4f}'.format(metrics.recall_score(Y_test, Y_pred, average='macro'))
  f1 = '{:02.4f}'.format(metrics.f1_score(Y_test, Y_pred, average='macro'))
  print(lat)
  result = model.predict([float(lat), float(lon)])
  performance = 'ACC:'+str(acc)+'; Precision:'+str(p)+'; Recall:'+ str(r)+'; F1-score:'+ str(f1)
  return [result[0], performance]

## Radius Searching Function
def rs(input, output, radius):
  df = pd.read_csv(input)
  lat00 = df['latitude']
  lon00 = df['longitude']
  iss = df['issue_question'].astype(str)
  size = len(lat00)
  isss = [x.lower() for x in iss]
  issr = ['']*size
  for i in range(0,size):
    lat0 = lat00[i]
    lon0 = lon00[i]
    for ii in range(0,size):
      dlon = lon00[ii] - lon0
      dlat = lat00[ii] - lat0
      a = (sin(dlat/2))**2 + cos(lat0) * cos(lat00[ii]) * (sin(dlon/2))**2
      c = 2 * atan2(sqrt(a), sqrt(1-a))
      distance = (3958.8) * c #miles
      if distance < radius: #searching radius in miles
        issr[i]=issr[i]+'/'+isss[ii]
  df1 = DataFrame(issr, columns= ['issue_question_r'])
  df = df.merge(df1, left_index = True, right_index = True)
  export_csv = df.to_csv (output, index = None, header=True)

## ====== web app links ======
## read index page
@app.route('/')
@app.route('/index.html')
def index():
    return render_template("index.html",
       title = 'Home', user = { 'nickname': 'company' },
       )
if __name__ == "__main__":
 	app.run(host="0.0.0.0", port=80)

## read resume page
@app.route('/resume')
def resume():
    return send_file("resume_Yu-Yen_Chang.pdf", as_attachment=True)

## read project page
@app.route('/project.html')
def project():
    return render_template("project.html")

## read blog page
@app.route('/blog.html')
def blog():
    return render_template("blog.html")

## read demo page
@app.route('/demo.html')
def demo():
    return render_template("demo.html")
## read input page
@app.route('/input')
def cesareans_input():
    return render_template("input.html")

## read map page
@app.route('/map')
def map():
    return render_template("map.html")

## read output page
@app.route('/output')
def cesareans_output():
  ## ====== inputs ======
  lat = request.args.get('lat')
  lon = request.args.get('lon')
  topic = request.args.get('topic')

  ## ====== radius searching =====
  # rs("company_topics_issues_topic0.csv", "company_topics_issues_text_topic0.csv", 10)

  ## ====== read data ======
  if (topic == 'Topic0'): df = pd.read_csv("company_topics_issues_text_topic0.csv") 
  issr = df['issue_question_r'].astype(str)
  size = len(issr)
  issr1 = np.zeros(size)

  ## ====== most common words ======
  # word=' '.join(list(issr))
  # allWords = nltk.tokenize.word_tokenize(word)
  # stopwords = nltk.corpus.stopwords.words('english')
  # allWordExceptStopDist = nltk.FreqDist(w.lower() for w in allWords if w not in stopwords)    
  # mostCommon= allWordExceptStopDist.most_common(100)
  # print(mostCommon)

  ## ====== find No. 1 issue ======
  for i in range(0,size):
    if (topic == 'Topic0'): 
      issr1a = issr[i].count('worda0') + issr[i].count('worda1') + issr[i].count('worda2') + issr[i].count('worda3')
      issr1b = issr[i].count('wordb0') + issr[i].count('wordb1') + issr[i].count('wordb2')
      issr1c = issr[i].count('wordc0') + issr[i].count('wordc1') + issr[i].count('wordc2')
      issr1d = issr[i].count('wordd0') + issr[i].count('wordd1') + issr[i].count('wordd2') + issr[i].count('wordd3')
      issr1[i] = np.flip(np.argsort([issr1a,issr1b,issr1c,issr1d]))[0]

  ## ====== fit the data ======
  result_issue = ml(lat, lon, df, issr1)
  # print(result_issue[1])

  ## ====== outputs ======
  if (topic == 'Topic0'): 
    if (result_issue[0] == 0): issueo = 'issue0'
    if (result_issue[0] == 1): issueo = 'issue1'
    if (result_issue[0] == 2): issueo = 'issue2'
    if (result_issue[0] == 3): issueo = 'issue3'

  the_result_1 = 'main issue: ' + issueo
  the_result_2 = topic + ' @ ('+ lat + ',' + lon +')'

  ## ====== heat map ======
  coord = list(zip(df['latitude'].tolist(), df['longitude'].tolist()))
  lat = request.args.get('lat')
  lon = request.args.get('lon')
  m = folium.Map([float(lat), float(lon)], zoom_start = 14, max_zoom = 15)
  m.add_child(plugins.HeatMap(coord, radius = 15))
  m.save('flaskexample/templates/map.html')

  ## ====== return to the output page ======
  return render_template("output.html", the_result_1 = the_result_1, the_result_2 = the_result_2)

