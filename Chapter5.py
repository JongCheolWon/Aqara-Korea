# 시리즈
# 리스트로 만들기 사례
import pandas as pd
sd1 = pd.Series(['Dog','Cat','Tiger','Lion','Monkey'], index = ['0','1','2','3','4'])

# 시리즈 데이터 출력
sd1


# 딕셔너리로 시리즈 만들기: pd.Series(딕셔너리)
# 라이브러리 불러오기
import pandas as pd

# 딕셔너리로 Series 생성
dict_data = {'a':1,'b':2,'c':3}
sd2 = pd.Series(dict_data)

# 타입 확인
type(sd2)

# 시리즈 데이터 출력
sd2


# 리스트로 시리즈 만들기: pd.Series(리스트)
import pandas as pd

# 리스트로 Series 생성
list_data = pd.Series(['Dog','Cat','Tiger','Lion','Monkey'], index=['0','1','2','3','4'])
sd3 = pd.Series(list_data)

# 데이터 확인
sd3


# 튜플로 시리즈 만들기: pd.Series(튜플)
import pandas as pd

# 튜플로 Series 생성
tup_data = ('이순신','1991-03-15','남')
sd4 = pd.Series(tup_data, index = ['이름','생년월일','성별'])

# 데이터 확인
sd4


# 데이터프레임
# 딕셔너리로 데이터프레임 만들기: pd.DataFrame(딕셔너리)
import pandas as pd

# 딕셔너리로 데이터프레임 생성
dict_data = {'동물':['Dog','Cat','Tiger','Lion','Monkey'],'나이':[7,9,2,3,1]}
df1 = pd.DataFrame(dict_data)

# 타입 확인
type(df1)

# 데이터 확인
print(df1)


# 리스트로 데이터프레임 만들기: pd.DataFrame(리스트)
import pandas as pd

# 리스트로 데이터프레임 생성
df2=pd.DataFrame([['Dog','7'],['Cat','9'],['Tiger','2'],['Lion','3'],['Monkey','1']], 
index = ['0','1','2','3','4'], columns = ['동물','나이'])

# 데이터 확인
df2


# 데이터프레임의 기본 정보 출력: df.info()
# 관련 라이브러리 불러오기
import pandas as pd
import seaborn as sns

# 데이터 준비 시본에 내장된 팁 데이터
tips = sns.load_dataset("tips")   # 팁 데이터
tips

# 데이터프레임의 기본 정보 확인
tips.info()


# 특정 행 상위 데이터 추출: df.head(n)
# 데이터프레임에서 상위 5개 데이터 추출
tips.head()

# 데이터프레임에서 상위 2개 데이터 추출
tips.head(2)


# 특정 행 하위 데이터 추출: df.tail(n)
# 데이터프레임에서 하위 데이터 추출
tips.tail()

# 데이터프레임에서 하위 3개 데이터 추출
tips.tail(3)


# 2번 인덱스를 삭제하고 새로운 데이터프레임 tips1 생성
tips1 = tips.drop([2])
tips1

# 행 번호 읽기로 세번째 데이터를 가지고 온다.
tips1.iloc[2]

# 2번 인덱스가 삭제되어 에러가 발생한다.
tips1.loc[2]


# loc로 행 데이터 추출하기: df.loc[인덱스 이름]
# 데이터프레임에서 프레임에서 인덱스가 0인 데이터 추출
tips.loc[0]

# 1, 3, 5 인덱스를 한번에 가져오기
tips.loc[[1,3,5]]


# iloc 속성으로 행 데이터 읽어오기: df.iloc[행 번호]
# 데이터프레임에서 0번째 행 데이터 추출
tips.iloc[0]


# iloc를 통해 마지막 행 데이터 가져오기: df.iloc[-1]
# -1을 사용하여 마지막 행 데이터를 추출한 예제
tips.iloc[-1]


# df[시작 행:마지막 행]
# 행 3개 가지고 오기
# tips[0:3]라고 하면 0, 1, 2번째 행을 출력한다. 데이터프레임의 첫 번째 행을 0번째 행이라고 가정하며
# [0:3]이라고 입력했지만 3번째 행을 가져오지 않음에 유의해야 한다.
tips[0:3]

# 행 번호를 이용하여 선택하기: .iloc
tips.iloc[0:3]

# 인덱스를 이용하여 선택하기: .loc
tips.loc[0:3]

# 인덱스 0부터 3까지의 컬럼 'sex'와 컬럼 'day' 가져오기
tips.loc[0:3,['sex','day']]

# 특정 인덱스 값의 컬럼 'sex'와 컬럼 'day' 가져오기
# .at을 이용할 수도 있다.
tips.at[0,'sex']

# 행과 열을 동시에 선택할 수 있다. 다음은 행 번호 기준으로 행과 열 데이터를 선택하는 예제이다. 
# 행 번호 3:5로 네 번째 행과 다섯 번째 행을 선택하며, 열은 0:2로 첫 번째 열과 두 번째 열을 선택한다.
tips.iloc[3:5,0:2]

# 인덱스 기준으로 행, 열을 지정하면 어떻게 될까?
# 행과 열의 인덱스를 리스트로 넘겨줄 수도 있다. 다음은 두 번째, 세 번째, 다섯 번째 행과, 첫 번째와 세 번째 열을 선택하는 예제
tips.iloc[[1,2,4],[0,2]]

# 명시적으로 행이나 열 선택 인자에 ':' 슬라이스를 전달하면 다음과 같이 행 또는 열 전체를 가져올 수도 있다.
tips.iloc[1:3,:]

tips.iloc[:,1:3]

# 값 하나를 선택하기 위해서는 특정 행과 열을 지정하는 방식으로 하면 된다. 아래의 두 방법 모두 동일한 방법이다.
tips.iloc[1,1]
tips.iat[1,1]


# 기본 조건식: and(&), or(|), not(~), 비교(==)
# 전체 데이터 중 팁이 5달러 이상
tips[tips.tip > 5]

# and 조건 - 남자 손님이면서 비흡연자
tips[(tips['sex']=='Male') & (tips['smoker']=='No')]


# df.isin(values)
# 일요일이면 True
[tips['day'].isin(['Sun'])]

# 팁을 1달러 지불한 고객
tips[tips['tip'].isin([1])]


# df.컬럼명 또는 df['컬럼명']
# 데이터프레임에서 tip과 size 가지고 오기
tips.tip

tips['size']


# df['컬럼명1','컬럼명2','컬럼명n']
# 데이터프레임 'total_bill', 'tip', 'day' 열 변수 추출
tips [['total_bill','tip','day']]


# 신규 df.컬럼명 + df.컬럼명
# 금액과 팁의 합계인 총액(total) 파생변수 만들기
tips['total'] = tips['total_bill'] + tips['tip']


# 자료형 확인: df.dtype, type(df.컬럼명 또는 df['컬럼명'])
# 자료형 변환: df.astype
# 데이터프레임 자료형 확인
tips.dtypes

# 데이터프레임 열의 자료형 확인
type(tips.total_bill)

# 데이터프레임 자료형 변환
# 카테고리형의 흡연 유무를 문자열로 변환하여 삽입
tips['smoker_str']=tips['smoker'].astype(str)


# df.count()
# 데이터프레임의 개수 확인
tips.count()

len(tips)


# 데이터프레임 인덱스 보기
tips.index

# 데이터프레임 컬럼 보기
tips.columns

# 행, 열 구조 보기
tips.values


# 데이터프레임 정렬: dataframe.sort_values()
# 튜플 정렬: sorted(tuple, key)
# 리스트 정령: list.sort(), sorted(list)
# '지급액' 열(by=' total_bill' )을 기준으로 index(axis=0) 오름차순 정렬하기
tips.sort_values(by=['total_bill'], axis=0)

# '지급액' 열(by=' total_bill ')을 기준으로 index(axis=0) 내림차순 정렬하기
# ascending=False 옵션 추가
tips.sort_values(by=['total_bill'], axis=0, ascending=False)

# 열 이름을 (알파벳 순서로) 정렬하기: axis=1
tips.sort_index(axis=1)


# df.concat()
# df.append()
# 시리즈를 새로운 행으로 연결한다.
import pandas as pd
s1 = pd.Series([0, 1], index=['A', 'B'])
s2 = pd.Series([2, 3, 4], index=['c', 'd', 'e'])
s1

s2

pd.concat([s1, s2])

# 옆으로 데이터 열을 연결하고 싶으면 axis=1로 인수를 설정한다.
df1 = pd.DataFrame([['Dog','3'], ['Bird','10'], ['Tiger','6'], ['Moose','3']], index = ['0','1','2','3'], columns = ['동물', '나이'])
df1

df2 = pd.DataFrame([['집','0'], ['초원','0'], ['수풀','0'], ['초원','1']], index = ['0','1','2','3'], columns = ['사는곳','뿔의 개수'])
df2

pd.concat([df1, df2], axis=1)


# df.merge()
import pandas as pd

# 병합용 데이터프레임 생성
df1 = pd.DataFrame({'고객번호': [1001, 1002, 1003, 1004, 1005, 1006, 1007], '이름': ['강감찬', '홍길동', '이순신', '장보고', '유관순', '신사임당', '세종대왕']}, columns=['고객번호', '이름'])
df1

df2 = pd.DataFrame({'고객번호': [1001, 1001, 1005, 1006, 1008, 1001], '금액': [10000, 20000, 15000, 5000, 100000, 30000]}, columns=['고객번호', '금액'])
df2

# inner join
pd.merge(df1, df2, on='고객번호')

# Full join 방식은 키 값이 한쪽에만 있어도 데이터를 출력
pd.merge(df1, df2, how='outer', on='고객번호')

# left, right 방식은 각각 첫 번째, 혹은 두 번째 데이터프레임의 키 값을 모두 보여준다.
pd.merge(df1, df2, how='left', on='고객번호')

pd.merge(df1, df2, how='right', on='고객번호')


# 범주형 빈도분석: df.value_counts()
# 범주형 교차분석: df.crosstab()
# 라이브러리 불러오기
import pandas as pd        # 판다스
import seaborn as sns      # 시본

# 데이터 준비
tips = sns.load_dataset("tips")     # 팁 데이터

# 데이터 보기
tips

# tips 데이터 유형 확인
# tips 데이터의 범주형(category) 변수는 sex, smoker, day, time 4개
tips.dtypes

# 성별 범주형 빈도분석
tips['sex'].value_counts()

# 요일별 범주형 빈도분석
tips['day'].value_counts()

# 성별, 요일별 교차분석
pd.crosstab(tips['sex'],tips['day'])

# 여백 또는 누적값 cumulatives
pd.crosstab(tips['sex'],tips['day']).apply(lambda r: r/len(tips), axis=1)


# 연속형 변수의 기술통계: df.describe()
# tips 데이터 유형 확인
# tips 데이터의 연속형(category) 변수는 total_bill, tip, size 3개
tips.dtypes

tips.describe()
