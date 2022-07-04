import pandas as pd
from sklearn import model_selection
from sklearn import metrics
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# 타이타닉 데이터 준비
file_path = 'C:/data/titanic.csv'

# read csv() 함수로 데이터프레임 변환
df = pd.read_csv(file_path)
df

## 데이터 전처리
# 불필요 컬럼 삭제 (x 변수)
del df['PassengerId']
del df['Name']
del df['Ticket']
del df['Cabin']

# 결측치 처리 (x)
df.dropna(thresh = int(len(df) * 0.5), axis = 1) # 결측치 제거
df['Embarked'] = df['Embarked'].fillna('S') # 최다 빈도 'S'로 대체
df['Age'] = df['Age'].fillna(df['Age'].mean()) # 나이 평균값으로 대체

# 범주형 컬럼 처리 (x)
df['Sex'] = df['Sex'].astype('category')
df['Pclass'] = df['Pclass'].astype('category')
df['Embarked'] = df['Embarked'].astype('category')
df = pd.get_dummies(df)

# 독립변수와 종속변수 구분 (x, y)
x_data = df.iloc[:,1:]
y_data = df.iloc[:,0]
x_data = x_data.values
y_data = y_data.values
print(x_data)
print(y_data)

## 머신러닝 작업 수행
# 데이터 분할(7:3)
x_train, x_test, y_train, y_test = model_selection.train_test_split(x_data, y_data, test_size=0.3)
# 로지스틱 모델링
estimator = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
estimator.fit(x_train, y_train)

# train 학습
y_predict = estimator.predict(x_train)
score = metrics.accuracy_score(y_test, y_predict)
print('test score: ', score)

# 모델 적용
print(x_test[:2])
y_predict = estimator.predict(x_test[:2])
print(y_predict)
for y1, y2 in zip(y_test, y_predict):
	print(y1, y2, y1==y2)


import pandas as pd
from sklearn import model_selection
from sklearn import metrics
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# 타이타닉 데이터 준비
file_path = 'C:/data/titanic.csv'

# read csv() 함수로 데이터프레임 변환
df = pd.read_csv(file_path)
df

## 데이터 전처리
# 불필요 컬럼 삭제 (x)
del df['PassengerId']
del df['Name']
del df['Ticket']
del df['Cabin']

# 결측치 처리 (x)
df.dropna(thresh=int(len(df) * 0.5), axis=1) # 결측치 제거
df['Embarked'] = df['Embarked'].fillna('S') # 최다 빈도 'S'로 대체
df['Age'] = df['Age'].fillna(df['Age'].mean()) # 나이 평균값으로 대체

# 범주형 컬럼 처리 (x)
df['Sex'] = df['Sex'].astype('category')
df['Pclass'] = df['Pclass'].astype('category')
df['Embarked'] = df['Embarked'].astype('category')
df = pd.get_dummies(df)

# 독립변수와 종속변수 구분 (x, y)
x_data = df.iloc[:,1:]
y_data = df.iloc[:,0]
x_data = x_data.values
y_data = y_data.values
print(x_data)
print(y_data)

## 머신러닝
# 데이터 분한(7:3)
x_train, x_test, y_train, y_test = model_selection.train_test_split(x_data, y_data, test_size=0.3)

# 의사결정나무 모델링
estimator = DecisionTreeClassifier(criterion='gini', max_depth=None, max_leaf_nodes=None, min_samples_split=2, min_samples_leaf=1, max_features=None)
estimator.fit(x_train, y_train)

# train 학습
y_predict = estimator.predict(x_train)
score = metrics.accuracy_score(y_train, y_predict)
print('train score: ', score)

# test 평가
y_predict = estimator.predict(x_test)
score = metrics.accuracy_score(y_test, y_predict)
print('test score: ', score)

# 모델 검증
print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict))

# 중요도
estimator.fit(x_train, y_train)
print(estimator.feature_importances_)

d = {'attribute': df.iloc[:,1:].columns, 'importance': estimator.feature_importances_}
df_importance = pd.DataFrame(d)
df_importance.sort_values('importance', ascending=False)
print(df_importance)

# 의사결정나무 그리기
# http://www.graphviz.org/download/
# pip install graphviz
