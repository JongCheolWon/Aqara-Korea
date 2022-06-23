# 관련 라이브러리 불러오기
import pandas as pd                   # 판다스
import matplotlib as mpl           # 맷플롯립
import matplotlib.pylab as plt     # 맷플롯립
import seaborn as sns              # 시본
import numpy as np                 # 넘파이

# 타이타닉 데이터 파일경로를 찾고 변수 file_path에 저장
file_path = 'C:/data/titanic.csv'

df = pd.read_csv(file_path) #read csv() 함수로 데이터프레임 변환
df

# 데이터 백업하기
titanic_copy_backup = df.copy()
# to_csv 메소드를 사용하여 내장 데이터프레임을 csv 파일로 지정
# C:\data 폴더에 저장
df.to_csv('C:/data/titanic_csv_backup.csv')

# 저장 파일 확인
file_path = 'C:/data/titanic_csv_backup.csv'
titanic_csv_backup = pd.read_csv(file_path)
titanic_csv_backup


df.info()  # 데이터프레임의 기본 정보 출력

df.describe() # 데이터프레임의 기초 통계 확인

df.types # 데이터 유형 확인

df.loc[0] # 데이터 행 인덱스 확인

df.head() # 상위 데이터 추출

df.tail() # 하위 데이터 추출

df.index # 데이터프레임 인덱스 보기

df.values # 행, 열 구조 보기

df.sort_values(by=['Fare'], axis=0) # 요금 기준 오름차순 정렬

df.sort_values(by=['Fare'], axis=0, ascending=False) # 요금 기준 내림차순 정렬

df.sort_index(axis=1) # 열 이름을 (알파벳 순서로) 정렬하기: axis=1 Age부터 출력

df.sum(axis=1) # 단순열 합계


df.count() # 데이터 개수 확인

df.isnull().sum() # 결측치 확인

# 결측치 시각화
sns.heatmap(df.isnull(), cbar=False) # 시본 시각화 확인

sns.heatmap(df.isnull(), cbar=True) # 시본 시각화 확인

import missingno as msno # missingno 패키지 임포트
msno.matrix(df) # 매트릭스 결측치 시각화

msno.bar(df) # 바차트 결측치 시각화

msno.heatmap(df) # 히트맵 결측치 시각화

msno.dendrogram(df) # 덴드로그램 결측치 시각화

# Cabin 결측 제거
df['Cabin'].value_counts() # Cabin 삭제 전 결측치 내용 확인

df = df.dropna(thresh=int(len(df) * 0.5), axis=1) # 결측치 제거
msno.matrix(df) # 결측치 제거 확인

# Embarked 대체
df['Embarked'].value_counts() # 범주형 개수 확인

df['Embarked'] = df['Embarked'].fillna('S') # 최다빈도 'S'로 대체

# 나이 대체
df['Age'] = df['Age'].fillna(df['Age'].mean()) # 나이 평균값으로 대체
msno.matrix(df) # 결측치 제거 최종 확인


# 데이터 바이닝 - 연속형 나이 변수 범주화 cut() 함수 사용
bins = [1, 20, 60, 100] # 1-20 : 미성년, 21-60 : 성년, 61-100 : 성년
df_Age = pd.cut(df["Age"], bins, labels = ["미성년", "성년", "노년"])
df["Age_class"] = df.Age
df

t1 = df.Age_class.unique() # 범주형 중복없이 value 추출
t2 = df[df.Age_class.isin(['미성년'])] # 원하는 데이터만
t3 = df[~df.Age_class.isin(['미성년'])] # 원하는 데이터 제외 필터링
print(t1)
print(t2)
print(t3)

# 전체를 3개 범주로 구분
df_Age2 = pd.qcut(df["Age"], q=3, labels = ["미성년", "성년", "노년"])
df["Age_class2"] = df_Age2
df

# 범주형 컬럼 처리
# 원핫인코딩
df['Sex'] = df['Sex'].astype('category')
df['Pclass'] = df['Pclass'].astype('category')
df['Embarked'] = df['Embarked'].astype('category')
df = pd.get_dummies(df)
df

# 성별, 선실, 나이 그룹에 의한 생존율을 데이터프레임으로 계산
# 행에는 성별 및 나이 그룹에 대한 다중 인덱스를 사용하고 열에는 선실 인덱스를 사용
df1 = df.groupby(['Sex', 'Age_class', 'Pclass'])["Survived"].mean()
df2 = df1.unstack("Pclass")
df2

# 피봇 테이블
# 성별 및 선실에 의한 생존율을 피봇 데이터 형태로 변환한다.
df.pivot_table(['Survived'], index=['Sex', 'Pclass'], aggfunc='mean')


import pandas_profiling
report = df.profile_report()
report.to_file('report.html')


# 범주형 확인
# 남녀 성비 확인
sns.catplot('Sex', kind='count', data=df)

# 좌석 등급별 성비 확인
sns.catplot('Pclass', kind='count', hue='Sex', data=df)

# 출발 항구별 분포 확인
sns.catplot('Embarked', kind='count', data=df)

# 연속형 # 나이 분포
df['Age'].hist()

# 연속형 # 요금 분포
df['Fare'].hist()


# 생존자 분석
sns.catplot('Survived', kind='count', hue='Sex', data=df)

# 보기 편하게 생존자 구분 변환
df['Survivor'] = df['Survived'].map({0 :'no', 1 : 'yes'})
# 생존/성별
sns.catplot('Survivor', kind='count', hue='Sex', data=df)
# 성별/생존자
sns.catplot('Sex', kind='count', hue='Survivor', data=df)
# 남자일수록 사망자가 많다.

# 좌석 등급별 생존자 확인
sns.catplot('Pclass', kind='count', hue='Survivor', data=df)

# 항구별/생존
sns.catplot('Embarked', kind='count', hue='Survivor', data=df)
# 생존율이 차이가 있음.

# 연령대별/사망자 20세 미만의 아이들을 확인
sns.catplot('Age', kind='count', hue='Survivor', data=df.loc[df['Age'] < 20])

# 연령대별/사망자 70세 이상 노인
sns.catplot('Age', kind='count', hue='Survivor', data=df.loc[df['Age'] >= 70])

# 20세 이상, 어른의 경우 연령에 따른 사망률과의 관계는 적으며, 남자일 경우 사망률이 높음.
sns.lmplot('Age', 'Survived', hue='Sex', data=df.loc[df['Age'] >= 20])

# Pairgrid 플롯 활용 범주 전체를 시각화 가능
# 생존률/성별, 객실 등급별, 탑승 항구별, 연령대 그룹별
Pg = sns.PairGrid(df, y_vars="Survived", x_vars=["Pclass", "Sex", "Embarked", "Age_class"], height=5, aspect=.7)
Pg.map(sns.pointplot, scale=1.3, errwidth=4, color="xkcd:plum")
Pg.set(ylim=(0, 1))
sns.despine(fig=Pg.fig, left=True)

# 한글 처리
plt.rc('font', family='Malgun Gothic')

# Pairgrid bar 서브플롯 적용
Pg = sns.PairGrid(df, y_vars="Survived", x_vars=["Pclass", "Sex", "Embarked", "Age_class"], height=5, aspect=.7)
Pg.map(sns.barplot)

# Pairgrid violin 서브플롯 적용
Pg.map(sns.violinplot)


# 성별/나이
sns.distplot(df['Age'].loc[df['Sex']=='male'])
sns.distplot(df['Age'].loc[df['Sex']=='female'])

# 연령에 따른 사망과 상관성이 성별에 따라 다르다.
# 남성일 경우 나이가 많을수록 사망자 수가 높지만, 여성의 경우 나이가 많을수록 사망자가 낮게 나타난다.

# 생존자/나이
sns.catplot('age', kind='count', hue='survivor', data=titanic)

# 여성의 경우 형제/자매가 많을수록 사망률이 증가하며, 남성의 경우 차이가 없는 경향이 있음.
sns.lmplot('SibSp', 'Survived', hue='Sex', data=df)

# 부모나 자식이 많은 가족의 경우 남자일 경우 사망자가 낮으며, 여성의 경우 사망자가 많은 경향이 있음.
sns.lmplot('Parch', 'Survived', hue='Sex', data=df)

# 파생변수 만들기
# 형제/자매와 부모/자식의 수를 더하여, 가족의 크기라는 특성을 생성
# 여성의 경우 가족 수가 많을수록 사망자가 감소하는 경향이 있으며, 남성의 경우 상관관계가 적음.

df['family_Size'] = df['SibSp'] + df['Parch']
sns.lmplot('family_Size', 'Survived', hue='Sex', data=df)

# 남녀 상관없이 요금의 경우 높을수록 생존율이 높음.
sns.lmplot('Fare', 'Survived', hue='Sex', data=df)


# 생존에 따른 bar 차트 함수
def bar_chart(feature):
	Survived = df[df['Survived']==1][feature].value_counts()
	Dead = df[df['Survived']==0][feature].value_counts()
	df_titanic = pd.DataFrame([Survived,Dead])
	df_titanic.index = ['Survived','Dead']
	df_titanic.plot(kind='bar', stacked=True, figsize=(10,5))

# 생존/성별
# 여성이 남성보다 생존할 가능성이 더 높다.
bar_chart('Sex')

# 생존/등급별
# 차트는 1등급 클래스가 다른 클래스보다 생존할 가능성이 더 높다.
# 차트는 3등급 클래스가 다른 클래스보다 죽을 가능성이 더 높다.
bar_chart('Pclass')

# 생존/항구별
bar_chart('Embarked')
