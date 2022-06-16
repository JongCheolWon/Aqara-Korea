# df.groupby(['그룹변수'])
# df.get_group('그룹변수지정')
import pandas as pd

file_path = 'C:/data/exam_sample.csv'
df = pd.read_csv(file_path)
df

# 반별로 그룹화(A,B,C반)
# 반별 그룹 오브젝트만 생성
df1 = df.groupby(['class'])
df1

# 반 중 A반 그룹만 확인
df1.get_group('A')

# 반별 그룹 평균 확인
df.groupby('class').mean()

# 반별, 성별 그룹 평균 확인
df.groupby(['class','sex']).mean()

# 반별 수학 평균
df['math'].groupby(df['class']).mean()
df.groupby(df['class'])['math'].mean()

# 반별 수학 개수
df['math'].groupby(df['class']).count()

# 성별 수학 평균
df_mean = df['math'].groupby(df['sex']).mean()
sexgroup = df.groupby('sex')
print(df_mean)
print(sexgroup)
print(sexgroup.groups)

# 남학생 수학 평균
male = sexgroup.get_group('m')
subset = male[['sex', 'math']]
print(male)
print(subset)


# pd.cut(): 동일 길이로 범주화
# pd.qcut(): 동일 개수로 범주화
# 판다스 라이브러리 불러오기
import pandas as pd

# 파일경로를 찾고 변수 file_path에 저장
file_path= 'C:/data/exam_sample.csv'

# read csv() 함수로 데이터프레임 변환
df = pd.read_csv(file_path)
df

# math 변수에 대해서 3개 동일한 길이로 범주형 변수로 만들어보겠다.
# 범주 구간은 Categories (3, interval[float64]): [(19.93, 43.333] < (43.333, 66.667] < (66.667, 90.0]]로 3개 구간 길이가 동일함을 알 수 있다.
df1 = pd.cut(df.math, 3)
df1

# math 변수에 3개 구간의 범주를 그룹 함수를 추가해서
# 각 범주의 그룹별로 agg() 함수인 개수(count), 평균(mean), 표준편차(std), 최솟값(min), 최댓값(max)을 계산해 보자
df2 = df.math.groupby(df1)
df3 = df2.agg(['count', 'mean', 'std', 'min', 'max'])
df3

# pd.qcut() 함수를 사용 math에 대해서 각 범주 구간별로 동일하게 3개의 개수를 가지도록 범주를 만들 수도 있다.
# 이때 labels=False로 설정하여 label이 0, 1, 2 구간을 0부터 순차적으로 1씩 증가하게 한다.
df4 = pd.qcut(df.math, 3, labels=False)
df4

# 아래처럼 labels=np.arange(3, 0, -1)로 직접 지정하면 label이 3, 2, 1로 3부터 1씩 줄어드는 순서로 할당이 된다. 위의 label과 정반대로 할당이 되었다.
import numpy as np
df5 = pd.qcut(df.math, 3, labels=np.arange(3, 0, -1))
df5

# [3 < 2 < 1] 순서로 동일 개수로 나눈 3개의 그룹별 통계량을 계산할 수 있다.
df6 = df.math.groupby(df5)
df7 = df6.agg(['count', 'mean', 'std', 'min', 'max'])
df7


# 범주형 자료 원-핫인코딩: pd.get_dummies(df)
# 유형 확인
df.dtypes

# 범주형 원-핫코딩
df = pd.get_dummies(df)
df


# 데이터 전치: df.T
# 원 데이터
df = pd.read_csv(file_path)
df

# 전치 데이터
df.T


# df.pivot(index, columns, values)
# df.pivot_table(data, index, columns, values, aggfunc)
# 데이터프레임을 확인
df

# 반(class) 변수를 행 데이터로 피봇
pd.pivot_table(df, index='class', columns='sex', values='science')

# 성별(sex) 변수를 행 데이터로 피봇
pd.pivot_table(df, index='sex', columns='class', values='science')


# pd.melt(df, id_vars=['id1','id2', ...])
# 멜트 함수
pd.melt(df,id_vars=['student_no', 'class'])


# 열 인덱스를 행 인덱스로 변환: stack()
# 행 인덱스를 열 인덱스로 변환: unstack()
# 스택
df5 = df.stack()
# 스택 후의 데이터프레임은 인덱스 레벨이 3개 있는 멀티인덱스(multiIndex)가 됨
df5

# 어떤 레벨이 컬럼으로 이동해서 언스택되는지 확인
# 스택으로 위에서 아래로 높게 올린 데이터프레임을(df5)를 가지고
# 거꾸로 왼쪽으로 오른쪽으로 넓게 언스택으로 펼쳐 보기
df6 = df.unstack(level=1)
df6


# Timestamp와 Period의 차이 확인, Timestamp를 Period로 변환
# Period 객체는 to_period(freq='기간인수')를 통해 datetime 변수에 대해 어떤 기간에 따른 자료형을 생성하고자 할 때 주로 활용
# datetime 유형에 대해서만 적용 가능

# 시간 정의
dates = ['2020-01-01','2020-03-01','2021-09-01']
dates

# 시간 자료형 생성
ts_dates = pd.to_datetime(dates)
ts_dates

# Timestamp를 Period 변환
pr_day = ts_dates.to_period(freq='D')      # 1일 기간
pr_day

pr_month = ts_dates.to_period(freq='M')    # 1개월 기간
pr_month

pr_year = ts_dates.to_period(freq='A')     # 1년 기간
pr_year


# pd.to_datetime(): 시계열 객체로 변환
# 판다스 라이브러리 불러오기
import pandas as pd
# 파일경로를 찾고 변수 file_path에 저장
file_path = 'C:/data/timeseries.csv'
# read csv() 함수로 데이터프레임 변환
df = pd.read_csv(file_path)

df

# 데이터 유형 확인
df.info()

# 현재 날짜를 나타내는 Date 컬럼은 문자형(object)이므로 to_datetime() 함수를 이용해서 Date 컬럼을 시계열 객체(Timestamp)로 변환한 후 확인
df['new_Date'] = pd.to_datetime(df['Date'])
df.info()

# 데이터프레임 확인
print(df.head())
print('\n')

print(df.info())
print('\n')
print(type(df['new_Date'][0]))

# 기존 Date 열을 삭제
df.drop('Date', axis = 1, inplace = True)

# new_Date를 인덱스로 지정
df.set_index('new_Date', inplace=True)
# 인덱스가 DatetimeIndex 변경되었고 2015년 07월 02일에서 2019년 06월 26일 사이에 5개 날짜가 있음

df

# 간단한 시계열 시각화
import matplotlib

df.plot


# pd.date_range(): 시작일과 종료일 또는 시작일과 기간을 입력하면 범위 내의 인덱스를 생성
# pd.period_range(): Period는 기간을 나타내는 자료형이며, 배열을 적용할 때 freq= 옵션은 기간의 단위를 의미
# 타임스탬프를 배열하는 date_range()는 파이썬 내장 함수 range() 함수와 비슷한 개념

import numpy as np
# pd.date_range()을 사용해 날짜 값들을 만들어 전달
dates = pd.date_range('2020010', periods=6)

# 컬럼의 이름은 A, B, C, D 라는 이름이 담긴 리스트에 추가
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
df

# 시계열 활용 날짜 데이터 분리: dt.year, dt.month, dt.day
import pandas as pd
file_path= 'C:/data/timeseries.csv'
df= pd.read_csv(file_path)
df

# 시간 변수 추가
df['new_Date'] = pd.to_datetime(df['Date'])
df

# 년, 월, 일 시간 추출
df['Year'] = df['new_Date'].dt.year
df['Month'] = df['new_Date'].dt.month
df['Day'] = df['new_Date'].dt.day
df

# to_period() 함수를 이용해 표기 변경
df['Date_yr'] = df['new_Date'].dt.to_period(freq = 'A')   # 연도까지
df['Date_m'] = df['new_Date'].dt.to_period(freq = 'M')    # 연월까지
df

# 날짜 인덱스 지정
df.set_index('new_Date', inplace=True)
df

# 날짜 인덱싱
df.loc['2015-07']  # 7월에 해당하는 row 인덱싱

df['2015-06-25' : '2018-06-20']  # 해당 기간 인덱싱

# 오늘 날짜와 차이 열 추가
today = pd.to_datetime('2020-03-18')
df['time_diff'] = today - df.index
df
