# NLP

## 1.데이터 획득
``` python
news = pd.read_json('./archive/News_Category_Dataset_v2.json', lines=True)
news = news[news['date'] >= pd.Timestamp(2018,1,1)]
```
해당 데이터 중 2018년 이상 존재하는 데이터들로만 사용하였습니다.

## 2. 데이터 전처리
중복되는 카테고리를 다음과 같은 함수를 사용하여 카테고리의 수를 줄입니다.
``` python
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
```
해당 데이터의 headline, short_description column을 하나로 합치고 다른 불 필요한 column들을 삭제하여 전처리를 진행하였습니다.


> "Stopword"
``` python
stop_words = set(stopwords.words('english'))
for i in range(len(news_articles_temp["information"])):
    string = ""
    for word in news_articles_temp["information"][i].split():
        word = ("".join(e for e in word if e.isalnum()))
        word = word.lower()
        if not word in stop_words:
            string += word + " "
    news["information"][i] = string.strip()
```

> "Lemmatizer"
```python
lemmatizer = nltk.WordNetLemmatizer()
for i in range(len(news_articles_temp["information"])):
    string = ""
    for w in nltk.word_tokenize(news_articles_temp["information"][i]):
        #print(w)
        string += lemmatizer.lemmatize(w, pos = "v") + " "
        #print(string)
    news["information"][i] = string.strip()
```
> information과 authors의 데이터를 합쳐서 하나의 벡터로 사용했습니다.
``` python
news['info_auth'] = news['information'] + news['authors']
```

## CountVectorizer를 통한 가장 빈도가 높은 단어 visulaization
``` python
temp_vect = sklearn.feature_extraction.text.CountVectorizer()
vect = temp_vect.fit(news['info_auth'])
count = vect.transform(news['info_auth']).toarray().sum(axis=0)
idx = np.argsort(-count)
count = count[idx]
feature_name = np.array(vect.get_feature_names())[idx]
plt.bar(feature_name[:10], count[:10])
plt.xticks(rotation=45)
plt.show()
```
![data_word](https://user-images.githubusercontent.com/49264688/121976381-a7f0f980-cdbe-11eb-9bab-a754cd7088d7.png)


## Data Category visualization
해당하는 데이터의 news category들과 분포를 pie 차트를 이용해서 볼 수 있다.
``` python
import plotly.graph_objs as go

labels = news['category'].value_counts().index
values = news['category'].value_counts().values

colors = news['category']
fig = go.Figure(data = [go.Pie(labels= labels, values= values, textinfo = "label+percent",
                               marker=dict(colors=colors))])
fig.show()
```
![카테고리 분포](https://user-images.githubusercontent.com/49264688/121976189-4597f900-cdbe-11eb-8260-2b2895b64211.PNG)


## TfidVectorizer를 사용하여 data vectorize, encoding, 
> "다음과 같은 argument 사용"
``` python
vectorize = sklearn.feature_extraction.text.TfidfVectorizer(analyzer = "word", ngram_range=(1, 2), min_df=2, max_df=0.5, smooth_idf=True)
encoder = LabelEncoder()
x = vectorize.fit_transform(news['info_auth'])
y = encoder.fit_transform(news['category'])
```

## 3. model training, testing 
``` python
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.3333, random_state=42)
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
```
모델 결과 스크린샷 입니다. 
![image](https://user-images.githubusercontent.com/49264688/121978631-8fcfa900-cdc3-11eb-932a-b610de8ab993.png)


위 결과로 LinearSVC_class_weight=balanced 가 가장 accuracy가 높은 것을 확인 할 수 있었습니다. 
## 4. Hyperparameter selection
위의 모델들을 사용하면서 Linear SVC가 가장 적절한 모델이라고 판단하여 선택한 후
argument C의 변화에 따른 정확도의 차이를 다음과 같이 구해서 C의 값을 선택 하였습니다.
C=1 을 기준으로 test error score가 점점 내려가는 것이 확인되어 C의 값은 1로 선택했습니다.
``` python
Cs = [0.001, 0.01, 1, 10, 100, 1000]

score1 = []
score2 = []

for C in Cs:
    model = LinearSVC(C=C)
    model.fit(X_train, Y_train)
    s1 = model.score(X_train, Y_train)
    s2 = model.score(X_test, Y_test)
    score1.append(s1)
    score2.append(s2)

plt.plot(score1, 'b^:')
plt.plot(score2, 'ro-')
plt.legend(['train', 'test'])
plt.xticks(range(len(Cs)), Cs)
```
![class_weight](https://user-images.githubusercontent.com/49264688/121976775-7a588000-cdbf-11eb-9bc2-c4173b239b00.png)

## 출처
https://www.kaggle.com/rmisra/news-category-dataset
kaggle news-category-dataset
https://www.kaggle.com/vikashrajluhaniwal/recommending-news-articles-based-on-read-articles
stopword, lemmatization
https://www.kaggle.com/parvezmullah/data-visualisation-and-accuracy-upto-73
data visualization 
