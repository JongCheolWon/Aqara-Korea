# 앤스콤 데이터는 seaborn 라이브러리에 내장되어 있음

# seaborn 라이브러리 호출
import seaborn as sns

# 앤스콤 데이터셋 로드
ans = sns.load_dataset('anscombe')

# 데이터 확인
ans

# 데이터 타입 보기
ans.dtypes

# 기초 통계량이 같더라도 데이터 분포는 다를 수 있다.

# dataset의 기초 통계량
ans.describe()

ans.groupby(['dataset']).describe()

# dataset 1 추출하여 그래프로 표현
import matplotlib.pyplot as plt

data1 = ans[ans['dataset']=='I']
data1

plt.plot(data1['x'],data1['y'])

plt.plot(data1['x'],data1['y'],'o')

# matplotlib로 그래프 그리기
# 1. 기본 틀 만들기(figure)
# 2. 내부에 격자를 만들고 서브 플롯을 생성(add_subplot)
# 3. 각 격자에 플롯을 그린다.
# 4. 타이틀 기본의 제목 등 부가적 작업을 수행

data2 = ans[ans['dataset']=='II']
data3 = ans[ans['dataset']=='III']
data4 = ans[ans['dataset']=='IV']

fig = plt.figure()

# 2*2 격자를 만들고 각 격자에 () 데이터셋의 플롯 그리기
# 기본 틀에 격자 추가

ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
fig

# 각 격자에 개별 데이터셋의 플롯 그리기
ax1.plot(data1['x'],data1['y'],'o')
ax2.plot(data2['x'],data2['y'],'o')
ax3.plot(data3['x'],data3['y'],'o')
ax4.plot(data4['x'],data4['y'],'o')

# 각 서브플롯의 타이틀을 추가
ax1.set_title('data1')
ax2.set_title('data2')
ax3.set_title('data3')
ax4.set_title('data4')

# 레이아웃 조절
fig.tight_layout()


import matplotlib as mpl
import matplotlib.pylab as plt

# 간단한 시작
import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4])
plt.show()


# 제목 추가
import matplotlib.pyplot as plt
plt.title('Line')
plt.plot([1,2,3], [1,2,3], marker='o')
plt.show()


# 범례 추가
plt.title('Legend')
plt.plot([1,2,3,4], label='asc')   # 증가
plt.plot([4,3,2,1], label='desc')  # 감소
plt.legend()
plt.show()


# 색상
plt.plot([1,2,3], [1,2,3], color='red')
plt.show()


# 축 이름
plt.plot([1,2,3], [1,2,3])
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# 선 모양 변경
plt.plot([1,2,3,4], color='r', linestyle='--', label='dashed') # 빨간색
plt.plot([4,3,2,1], color='g', linestyle=':', label='dotted')  # 감소
plt.legend()
plt.show()


# 그림 범위
plt.title("x, y Range")
plt.plot([10, 20, 30, 40], [1, 4, 9, 16], c="b", lw=5, ls="--", marker="o", ms=15, mec="g", mew=5, mfc="r")
plt.xlim(0, 50)
plt.ylim(-10, 30)
plt.show()


# df.plot(x=변수, y=변수, kind 옵션)
# 라이브러리 불러오기
import pandas as pd    # 판다스
import seaborn as sns  # 시본
# 데이터 준비
tips = sns.load_dataset("tips")  # 팁 데이터

# 데이터 보기
tips

# 기본 선
tips.plot()
tips.plot(kind='line')

# 히스토그램
tips.plot(kind='hist')

# 박스플롯
tips.plot(kind='box')


# 관련 라이브러리 불러오기
import pandas as pd               # 판다스
import matplotlib as mpl          # 맷플롯립
import matplotlib.pylab as plt    # 맷플롯립
import seaborn as sns             # 시본
import numpy as np                # 넘파이

# 데이터 준비
tips = sns.load_dataset("tips")   # 팁 데이터

# 단순그래프
plt.plot(tips.total_bill)


# X축 범주형, Y축 연속형
plt.bar(tips.sex, tips.total_bill)


# X축 연속형, Y축 연속형
plt.scatter(tips.total_bill, tips.tip)


# X축 변수. Y축 빈도
plt.hist(tips.total_bill)

plt.hist(tips.time)


import pandas as pd
import matplotlib as mpl
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np

# 한글 타이틀
plt.rc('font', family='Malgun Gothic')

# 데이터 준비
iris = sns.load_dataset("iris")         # 붓꽃 데이터
titanic = sns.load_dataset("titanic")   # 타이타닉호 데이터
tips = sns.load_dataset("tips")         # 팁 데이터
lights = sns.load_dataset("flights")    # 여객 운송 데이터

tips

# 기본 배경 설정
sns.set_palette("pastel")

# 시각화
sns.stripplot(x="day", y="total_bill", data=tips)

# 개인화
plt.title('팁데이터')
plt.ylabel("요금")
plt.xlabel("요일")


# 변수: sns.barplot(X,Y,data)
# X축 데이터프레임 변수, Y축 데이터프레임 변수, data: 데이터프레임
# 단일변수: sns.catplot('변수', kind='count'데이터프레임)
# 평균치 빠르게 집계해서 시각화하기
sns.barplot(x="sex", y="tip", data=tips)

# estimator를 지정(여기에 사용할 값으로는 list에 해당하는 함수를 사용할 수 있음)
sns.barplot(x="sex", y="tip", estimator=len, data=tips)

# 여러 열에서 집단별로 그룹하고 세부 집단 시각화하기
# hue 파라미터 추가
sns.barplot(x="sex", y="tip", hue="day", data=tips)

# 표준편차 오차 막대
sns.barplot(x="sex", y="tip", hue="day", data=tips, ci="sd")

# 요일별 팁 카운트
sns.catplot('day', kind='count', data=tips)


sns.boxplot(x="sex", y="tip", data=tips)

sns.boxplot(data=tips, orient="h")


# 박스플롯 활용 이상치 제거 분석
tips

sns.boxplot(data=tips)

tips.describe()

# total_bill 40 이하 tip 8 데이터만 추출 - 조건에 맞는 불린 값 추출
(tips["total_bill"] < 40) & (tips["tip"] < 8)

# 데이터프레임으로 구성하여 확인하고 새로운 변수 newtips 구성
# 조건에 해당하는 10행 데이터가 제거되었다.
newtips=tips[(tips["total_bill"] < 40) & (tips["tip"] < 8)]
newtips

# 이상치 조정 후 시각화 결과 확인
sns.boxplot(data=newtips)


import plotly.graph_objects as go

fig = go.Figure(data=go.Countour(z=[[10, 10.625, 12.5, 15.625, 20], [5.625, 6.25, 8.125, 11.25, 15.625], [2.5, 3.125, 5., 8.125, 12.5], [0.625, 1.25, 3.125, 6.25, 10.625], [0, 0.625, 2.5, 5.625, 10]]))
fig.show()

#인터랙티브 기능 확인


# 라이브러리 불러오기
import pygal
# 그래프 생성
bar_chart = pygal.Bar()
# 값 추가
bar_chart.add('Fibonacci', [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55])
# svg 출력 파일 확인
bar_chart.render_to_file('bar_chart.svg')


# 보케
import bokeh.sampledata
# Standard imports
from bokeh.io import output_notebook, show
output_notebook()
# Plot a complex chart with intearctive hover in a few lines of code
# 코드 참조


# 홀로뷰
from itertools import islice
import numpy as np
import holoviews as hv
from holoviews import opts


from dateutil.parser import parse
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
plt.rcParams.update({'figure.figsize' : (10, 7), 'figure.dpi' : 120})

# 판다스 라이브러리 불러오기
import pandas as pd

# 파일경로를 찾고 변수 file_path에 저장
file_path = 'C:/data/time_series.csv'

# read csv() 함수로 데이터프레임 변환
df = pd.read_csv(file_path)

print(df)
# 현재 날짜를 나타내는 Date 칼럼은 문자형(object)이므로 to_datetime() 함수를 이용해서 Date 컬럼을 시계열 객체(Timestamp)로 변환

df.info()

df['new_date'] = pd.to_datetime(df['date'])
print(df.head())
print('\n')
print(df.info())
print('\n')
print(type(df['new_date'][0]))
df.drop('date', axis=1, inplace=True)
df.set_index('new_date', inplace=True)
print(df.head())
print('\n')
print(df.info())

from dateutil.parser import parse
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
plt.rcParams.update({'figure.figsize' : (10, 7), 'figure.dpi' : 120})

# Draw Plot
def plot_df(df, x, y, title="", xlabel='Date', ylabel='Value', dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

plot_df(df, x=df.index, y=df.value, title='Time series data')

# Import Data
df = pd.read_csv('c:/data/time_series.csv', parse_dates=['date'], index_col='date')
df.reset_index(inplace=True)

# Prepare data
df['year'] = [d.year for d in df.date]
df['month'] = [d.strftime('%b') for d in df.date]
years = df['year'].unique()

# Prep Colors
np.random.seed(100)
mycolors = np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), len(years), replace=False)

# Draw Plot
plt.figure(figsize=(16,12), dpi=80)
for i, y in enumerate(years):
	if i > 0:
		plt.plot('month', 'value', data = df.loc[df.year==y, :], color = mycolors[i], label=y)
		plt.text(df.loc[df.year==y, :].shape[0]-.9, df.loc[df.year==y, 'value'][-1:].values[0], y, fontsize=12, color=mycolors[i])

# Decoration
plt.gca().set(xlim=(-0.3, 11), ylim=(2, 30), ylabel='$Drug Sales$', xlabel='$Month$')
plt.yticks(fontsize=12, alpha=.7)
plt.title("Time series data", fontsize=20)
plt.show()

# Import Data
df = pd.read_csv('c:/data/time_series.csv', parse_dates=['date'], index_col='date')
df.rest_index(inplace=True)

# Prepare data
df['year'] = [d.year for d in df.date]
df['month'] = [d.strftime('%b') for d in df.date]
years = df['year'].unique()

# Draw Plot
fig, axes = plt.subplots(1, 2, figsize=(20, 7), dpi=80)
sns.boxplot(x='year', y='value', data=df, ax=axes[0])
sns.boxplot(x='month', y='value', data=df.loc[~df.year.isin([1991, 2008]), :])

# Set Title
axes[0].set_title('Year-wise Box Plot\n(The Trend)', fontsize=18);
axes[1].set_title('Month-wise Box Plot\n(The Seasonality)', fontsize=18)
plt.show()
