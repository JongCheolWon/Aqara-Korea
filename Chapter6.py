# isnull(): 결측 데이터이면 True 값을 반환하고, 유효한 데이터가 존재하면 False를 반환
# notnull(): 유효한 데이터가 존재하면 True를 반환하고, 누락 데이터이면 False를 반환
import pandas as pd

file_path = 'C:/data/bicycle.csv'
df = pd.read_csv(file_path, engine='python')
df

# 결측 데이터 확인
df.isnull()

df.notnull()


# 컬럼별 결측값 개수: df.isnull().sum()
# 행(row) 단위로 결측값 개수: df.isnull().sum(1)
# 행(row) 단위로 실제값 개수: df.notnull().sum(1)
# 결측값 개수
df.isnull().sum()

# 행 단위 결측값 개수
df.isnull().sum(1)

# 행 단위 실제값 개수
df.notnull().sum(1)


# 행 삭제: df.dropna(axis=0)
# 열 삭제: df.dropna(axis=1)
# df.dropna()
# 결측 데이터가 있는 전체 행 제거
df_drop_allrow = df.dropna(axis=0)
df_drop_allrow

# 결측 데이터가 있는 전체 열 제거
df_drop_allcolumn = df.dropna(axis=1)
df_drop_allcolumn

# 특정 행 또는 열 결측치 제거, 대여소번호 컬럼 제거(비교 확인)
df['대여소번호'].dropna()

# 결측값이 들어있는 행 전체 삭제: 하단 df.dropna(axis=0)와 동일
df[['대여소번호','대여거치대','이용시간']].dropna()
df[['대여소번호','대여거치대','이용시간']].dropna(axis=0)

# 결측값이 들어있는 열 전체 삭제
df[['대여소번호','대여거치대','이용시간']].dropna(axis=1)


# 결측값을 특정 값으로 대체: df.fillna(0)
# 결측값을 특정 문자열로 대체: df.fillna('')
# 결측값을 변수별 평균으로 대체: df.fillna(df.mean())
# 결측값을 특정 값(0)으로 대체
df_1 = df.fillna(0)
df_1

# 특정 컬럼 결측값을 특정 값(0)으로 대체
df_2 = df.대여소번호.fillna(0)
df_2

# 결측값을 문자열('missing')로 대체
df_3 = df.fillna('missing')
df_3

# 각 컬럼의 평균을 구해서 대응하는 컬럼의 결측값을 대체하는 방법으로 가장 많이 사용한다.
df_4 = df.fillna(df.mean())
df_4

# 특정 항목 평균 구하기(이용거리 평균)
df_5 = df.mean()['이용거리']
df_5

# 특정 항목 평균으로 대체
df_6 = df.fillna(df.mean()['이용거리'])
df_6

## 특정 항목 평균으로 대체
df_7 = df.이용거리.fillna(df.mean()['이용거리'])
df_7


import pandas as pd

# 파일경로를 찾고 변수 file_path에 저장
file_path = 'C:/data/bicycle_out.csv'

# read csv() 함수로 데이터프레임 변환
df = pd.read_csv(file_path, engine='python')
df

# 이상값이 있는 4번째 행 제거
df1 = df.drop(4,0)
df1


# 고급 시각화 박스플롯 참조
# 이용시간 이상치 시각화
import matplotlib as mpl         # 맷플롯립
import matplotlib.pylab as plt   # 맷플롯립
plt.boxplot(df['나이'])
plt.show()


# 고급 시각화 박스플롯 참조
# 이용시간 이상치 시각화
import matplotlib as mpl         # 맷플롯립
import matplotlib.pylab as plt   # 맷플롯립
plt.boxplot(df['나이'])
plt.show()


# Keep = 'first', 'last', False
# 중복이 있으면 처음이나 끝에 무엇을 남길지 확인한다.
# keep = 'first'가 default이며,
# 중복값이 있으면 첫 번째 값을 duplicated 여부를 False로 반환, 나머지 중복값에 대해서는 True를 반환
df.duplicated(['이용거리'],keep='first')

# keep = 'last'는 중복값이 있으면 첫 번째 값을 duplicated 여부를 True로 반환, 나머지 중복값에 대해서는 False를 반환
df.duplicated(['이용거리'],keep='last')

# keep = False는 처음이나 끝 값인지 여부는 고려를 안 하고 중복이면 무조건 True를 반환
df.duplicated(['이용거리'],keep=False)


# 데이터프레임.drop_duplicates()
# drop_duplicates()는 중복값을 keep='first', 'last', 'False argument에 따라 유일한 1개의 key 값만 남기고 나머지는 중복 제거
df.drop_duplicates(['이용거리'],keep='first')

df.drop_duplicates(['이용거리'], keep='last')

df.drop_duplicates(['이용거리'], keep=False)
