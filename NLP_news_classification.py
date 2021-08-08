
import pandas as pd
from nltk.corpus import stopwords
import numpy as np
import nltk

import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터 획득
news = pd.read_json('./archive/News_Category_Dataset_v2.json', lines=True)
news = news[news['date'] >= pd.Timestamp(2018,1,1)]

# 2. 데이터 전처리
def category_merge(x):
    if x == 'THE WORLDPOST':
        return 'WORLDPOST'
    elif x == 'TASTE':
        return 'FOOD & DRINK'
    elif x == 'STYLE':
        return 'STYLE & BEAUTY'
    elif x == 'PARENTING':
        return 'PARENTS'
    elif x == 'COLLEGE':
        return 'EDUCATION'
    elif x == 'ARTS' or x == 'CULTURE & ARTS':
        return 'ARTS & CULTURE'
    else:
        return x
news['category'] = news['category'].apply(category_merge)
news['information'] = news[['headline', 'short_description']].apply(lambda x: ' '.join(x), axis=1)
pd.set_option('display.max_colwidth', -1)

news.drop(news[(news['authors'] == '') & (news['short_description'] == '' )].index, inplace=True)

news_articles_temp = news.copy()

#stop words와 lemmatizer 설정
stop_words = set(stopwords.words('english'))
for i in range(len(news_articles_temp["information"])):
    string = ""
    for word in news_articles_temp["information"][i].split():
        word = ("".join(e for e in word if e.isalnum()))
        word = word.lower()
        if not word in stop_words:
            string += word + " "
    news["information"][i] = string.strip()
#
lemmatizer = nltk.WordNetLemmatizer()
for i in range(len(news_articles_temp["information"])):
    string = ""
    for w in nltk.word_tokenize(news_articles_temp["information"][i]):
        #print(w)
        string += lemmatizer.lemmatize(w, pos = "v") + " "
        #print(string)
    news["information"][i] = string.strip()

news['info_auth'] = news['information'] + news['authors']
# CountVectorizer를 통한 가장 많이 나오는 단어 visualization
temp_vect = sklearn.feature_extraction.text.CountVectorizer(stop_words='english')
vect = temp_vect.fit(news['info_auth'])
count = vect.transform(news['info_auth']).toarray().sum(axis=0)
idx = np.argsort(-count)
count = count[idx]
feature_name = np.array(vect.get_feature_names())[idx]
plt.bar(feature_name[:10], count[:10])
plt.xticks(rotation=45)
plt.show()

# Data category visualization
import plotly.graph_objs as go

labels = news['category'].value_counts().index
values = news['category'].value_counts().values

colors = news['category']
fig = go.Figure(data = [go.Pie(labels= labels, values= values, textinfo = "label+percent",
                               marker=dict(colors=colors))])
fig.show()


# TfidfVectorizer를 사용하여 data vectorize, encoding
vectorize = sklearn.feature_extraction.text.TfidfVectorizer(analyzer = "word", ngram_range=(1, 2), min_df=2, max_df=0.5, smooth_idf=True)
encoder = LabelEncoder()
x = vectorize.fit_transform(news['info_auth'])
y = encoder.fit_transform(news['category'])
# X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(vectorize.fit_transform(news['headline']), encoder.fit_transform(news['category']), test_size=0.33)
# X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(vectorize.fit_transform(news['information']), encoder.fit_transform(news['category']), test_size=0.33)
# X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(vectorize.fit_transform(news['short_description']), encoder.fit_transform(news['category']), test_size=0.33)
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.3333, random_state=42)

# 3. Model training, testing
from sklearn.svm import LinearSVC
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import ensemble, tree
from sklearn.neural_network import MLPClassifier

models = [
    {'name': 'LinearSVC', 'obj': LinearSVC()},
    {'name': 'LinearSVC_loss=hinge', 'obj': LinearSVC(loss='hinge')},
    {'name': 'LinearSVC_C=1', 'obj': LinearSVC(C=1)},
    {'name': 'LinearSVC_class_weight=balanced', 'obj': LinearSVC(class_weight='balanced')},
    {'name': 'LinearSVC_hinge,balanced', 'obj': LinearSVC(loss='hinge', class_weight='balanced')},
    {'name': 'LinearSVC_balanced_random', 'obj': LinearSVC(class_weight='balanced', random_state=1)},
    {'name': 'LogisticRegression', 'obj': LogisticRegression()},
    {'name': 'MultinomialNB_alpha=0.1', 'obj': MultinomialNB(alpha=0.1)},
    {'name': 'MultinomialNB', 'obj': MultinomialNB()},
    {'name': 'RandomForest', 'obj': ensemble.RandomForestClassifier(n_estimators=10)},
    {'name': 'AdaBoostClassifier', 'obj': ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier())},
    {'name': 'RidgeClassifier', 'obj': linear_model.RidgeClassifier(alpha=0.1)},
    {'name': 'linear_model.SGD', 'obj': linear_model.SGDClassifier()},
    {'name': 'neural_network.MLP', 'obj': MLPClassifier()},
]
print('=====News Category Classification=====')
for model_dict in models:
    model = model_dict['obj']
    model.fit(X_train, Y_train)
    Y_predict = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_predict) * 100
    # accuracy = metrics.balanced_accuracy_score(Y_test, Y_predict)
    # print('===== Result =====')
    print(f'model_name : {model_dict["name"]}, accuracy : {accuracy:.4}')
    # print(f'accuracy : {accuracy:.4}')
    # print('')

# 4. Hyperparameter selection
Cs = [0.001, 0.01, 1, 10, 100, 1000]

score1 = []
score2 = []

for C in Cs:
    model = LinearSVC(C=C, class_weight='balanced')
    model.fit(X_train, Y_train)
    s1 = model.score(X_train, Y_train)
    s2 = model.score(X_test, Y_test)
    score1.append(s1)
    score2.append(s2)

plt.plot(score1, 'b^:')
plt.plot(score2, 'ro-')
plt.legend(['train', 'test'])
plt.xticks(range(len(Cs)), Cs)


plt.show()


