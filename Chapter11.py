# 각 변수 간의(열 간의) 상관관계를 보려면 corr() 함수
# 판다스 라이브러리 불러오기
import pandas as pd
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns

# 데이터 불러오기
file_path = 'C:/data/exam_sample_cor.csv'

# read csv() 함수로 데이터프레임 변환
df = pd.read_csv(file_path)
df

# 두 연속형 변수 스캐터플롯 그리기
plt.scatter(df.science, df.math)

# 상관계수 확인
corr = df.corr() #(method = 'pearson')
corr

# 히트맵으로 상관도 시각화
sns.heatmap(data = df.corr(), annot=True, fmt = '.2f', linewidths=.5, cmap='Blues')

df1 = df.corr()
# 그림 사이즈 지정
fig, ax = plt.subplots(figsize=(7,7))

# 삼각형 마스크를 작성 (위 삼각형 True, 아래 삼각형 False)
mask = np.zeros_like(df1, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# 히트맵 그리기
sns.heatmap(df1, cmap = 'RdYlBu_r', annot = True, mask = mask, linewidths = 5, cbar_kws = {"shrink": .5}, vmin = -1, vmax = 1)
# annot = True : 실젯값을 표시한다. mask=mask : 표시하지 않을 마스크 부분을 지정한다. linewidth = 5 : 경계면 실선으로 구분하기. cbar_kws={"shrink":.5} : 컬러바 크기 절반으로 줄이기. vmin=-1, vmax=1 : 컬러바 범위 -1~1

df1 = df.corr()
sns.clustermap(df1, annot = True, cmap = 'RdYlBu_r', vmin = -1, vmax = 1)
# annot = True : 실젯값 화면에 나타내기. cmap = 'RdYlBu_r' : Red, Yellow, Blue 색상으로 표시. vmin = -1, vmax = 1 : 컬러차트 -1~1 범위로 표시


# 시본으로 과학, 수학 점수를 상관분석해 보자.
sns.lmplot(x='science', y='math', data=df)

# 회귀분석을 위해 종속(Y=수학), 독립(X=과학)하는 모델을 만든다.
# 과학 점수를 알면 수학 점수를 예상할 수 있다.
# 단순선형회귀 모형
import statsmodels.api as sm
lin_reg = sm.OLS.from_formula("math ~ science", df).fit()
lin_reg.summary()


import pandas as pd    # 판다스
import seaborn as sns  # 시본
import numpy as np     # 넘파이

# DataFrame 생성
passtest = [0,0,0,0,0,1,1,1,1,1]
score = [51, 64, 60, 50, 68, 80, 90, 92, 99, 83]
df = pd.DataFrame({"passtest": passtest, "score": score, })
df.head()

# 상관분석
sns.lmplot(x='score', y='passtest', data=df, logistic=True)


# y=ax+b에서 a는 1, b는 0으로 가정
# %matplotlib inline
import numpy as np # 넘파이 사용
import matplotlib.pyplot as plt # 맷플롯립 사용
def sigmoid(x):
		return 1/(1+np.exp(-x))
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x, y, 'g')
plt.plot([0,0],[1,0,0,0], ':') # 가운데 점선 추가
plt.title('Sigmoid Function')
plt.show()


def sigmoid(x):
    return 1/(1+np.exp(-x))
x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(0.5*x)
y2 = sigmoid(x)
y3 = sigmoid(2*x)

plt.plot(x, y1, 'r', linestyle='--') # W의 값이 0.5일때
plt.plot(x, y2, 'g') # W의 값이 2일때
plt.plot(x, y3, 'b', linestyle='--') # W의 값이 2일때
plt.plot([0,0],[1.0,0.0], ':') # 가운데 점선 추가
plt.title('Sigmoid Function')
plt.show()


def sigmoid(x):
    return 1/(1+np.exp(-x))
x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x+0.5)
y2 = sigmoid(x+1)
y3 = sigmoid(x+1.5)

plt.plot(x, y1, 'r', linestyle='--') # x + 0.5
plt.plot(x, y2, 'g') # x + 1
plt.plot(x, y3, 'b', linestyle='--') # x + 1.5
plt.plot([0,0],[1.0,0.0], ':') # 가운데 점선 추가
plt.title('Sigmoid Function')
plt.show()
