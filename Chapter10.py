import numpy   # 표본 집단을 랜덤 값으로 생성하기 위해서 사용
from scipy import stats   # t검정 수행을 위한 패키지 (내장)

# 학생 키에 대한 리스트 구성

height_list = numpy.array([169, 167, 175, 166, 162, 180, 172, 162, 173, 162, 181, 175, 181, 162, 165, 172, 176, 167, 165])
height_list

# T-검정 수행

# 귀무가설이 참인 경우 학생들의 평균 키는 170cm와 차이가 없다. 그러므로 평균 키는 170cm이다. (P>0.05)

# 귀무가설이 거짓인 경우 학생들의 평균 키는 170cm와 차이가 있다. 그러므로 평균 키는 170cm가 아니다. (P<0.05)

tTestResult = stats.ttest_1samp(height_list, 170)
tTestResult

# 결과 출력

# 귀무가설이 맞거나 틀린 것을 증명하려면 어떤 증거가 있어야 한다. 이 증거에 해당하는 숫자를 검정 통계량이라고 한다.

print('t검정 통계량 = %.3f, pvalue = %.3f'%(tTestResult))

# t검정 통계량 = 0.423, pvalue = 0.677


import numpy     # 표본 집단을 랜덤 값으로 생성하기 위해서 사용
from scipy import stats   # t검정 수행을 위한 패키지 (내장)

group1Heights = numpy.array([162, 168, 169, 165, 166, 168, 162, 172, 157, 173, 158, 169, 164, 170, 163, 175, 177, 162, 175, 177])

group2Heights = numpy.array([180, 181, 163, 164, 174, 169, 164, 172, 162, 171, 180, 168, 164, 169, 169, 178, 177, 167, 179, 172])

tTestResult = stats.ttest_ind(group1Heights, group2Heights)
tTestResult

# Ttest_indResult(statistic=-1.8253194633881713, pvalue=0.07582039848129221)


import numpy      # 표본 집단을 랜덤 값으로 생성하기 위해서 사용
from scipy import stats    # t검정 수행을 위한 패키지 (내장)

# 복용 전 몸무게에 대한 데이터
beforeWeights = numpy.array([80, 82, 76, 82, 65, 73, 77, 78, 61, 81, 80, 70, 60, 83, 89, 84, 85, 81, 67, 60])

# 복용 후 몸무게를 가정한 데이터
afterWeights = numpy.array([78.22687381, 79.5718219, 71.46930023, 81.04666603, 64.27984382, 72.41880152, 76.62206947, 78.10407236, 60.8858263, 83.51855868, 78.70976324, 68.15053364, 59.15290613, 80.47457326, 87.79704858, 83.03546344, 86.71565253, 78.13118404, 66.79769321, 61.59242483])

tTestResult = stats.ttest_rel(beforeWeights, afterWeights)
tTestResult

# Ttest_relResult(statistic=3.278149526008354, pvalue=0.003955230626284828


import numpy      # 표본 집단을 랜덤 값으로 생성하기 위해서 사용
from scipy import stats    # t검정 수행을 위한 패키지 (내장)

# 복용 전 몸무게에 대한 데이터
beforeWeights = numpy.array([80, 82, 76, 82, 65, 73, 77, 78, 61, 81, 80, 70, 60, 83, 89, 84, 85, 81, 67, 60])

# 복용 후 몸무게를 가정한 데이터
afterWeights = numpy.array([78.22687381, 79.5718219, 71.46930023, 81.04666603, 64.27984382, 72.41880152, 76.62206947, 78.10407236, 60.8858263, 83.51855868, 78.70976324, 68.15053364, 59.15290613, 80.47457326, 87.79704858, 83.03546344, 86.71565253, 78.13118404, 66.79769321, 61.59242483])

tTestResult = stats.ttest_rel(beforeWeights, afterWeights)
tTestResult

# Ttest_relResult(statistic=3.278149526008354, pvalue=0.003955230626284828
