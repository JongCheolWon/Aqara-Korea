# 결과 출력을 위해 Print 함수를 사용
# 기본 연산
print(123 + 456)
print(100 + 1004)

print('\n')

# 정수형 상수의 표현 범위는 제한이 없음
# CPU 레지스터로 표현할 수 있는 크기보다 큰 정수를 다룰 때는 연산 속도가 느려짐
print(2 ** 64)

print('\n')

# 정확히 나누어 떨어지지 않는 경우, 파이썬에서는 소수점 15자리까지 표현
print(5048 / 123)

print('\n')

# 소수점 이하의 수를 모두 버리고 몫만 나타낼 때 '//' 연산자를 사용
print(5048 // 123)

print('\n')

# 나머지를 구할 때 '%' 연산자를 사용
print(33 % 5)

print('\n')

print("===================================")

# 논리 연산
# and
print(True and True)
print(True and False)
print(False and True)
print(False and False)

print('\n')

#or
print(True or True)
print(True or False)
print(False or True)
print(False or False)

print('\n')

#not
print(not True)
print(not False)

print('\n')

print("===================================")

# 비교 연산
# 비교 판단
print(5 > 2)

print('\n')

# 숫자 비교
print(7 == 7)
print(7 != 5)

print('\n')

# 문자 비교
print('Python' == 'Python')
print('Python' == 'python')
print('Python' != 'python')

print('\n')

# 부등호 비교
print(1 > 2) # 1이 2보다 큰지 비교
print(1 < 2) # 1이 2보다 작은지 비교
print(1 >= 1) # 1이 1보다 크거나 같은지 비교
print(1 <= 1) # 1이 1보다 작거나 같은지 비교

print('\n')

# 객체 비교
print(1 == 1.0)
print(1 is 1.0)
print(1 is not 1.0)

print('\n')

print("===================================")

# 변수 만들기(할당)
x = 155
print(x)

print('\n')

y = 'Hello, Python!'
print(y)

print('\n')

# 여러 변수 만들기(할당)
## 변수명1, 변수명2, 변수명3 = 값1, 값2, 값3 형식으로 변수를, (콤마)로 구분한 뒤 각 변수에 할당될 값을 지정
## 변수와 값의 개수는 동일하게 하고 나열된 순서대로 값이 할당되며 만약 변수와 값의 개수가 맞지 않으면 에러가 발생
x, y, z = 100, 200, 300
print(x)
print(y)
print(z)

print('\n')

# 변수 삭제(메모리 할당 해제)
# 변수 삭제는 del을 사용
# 변수 x를 삭제하여 변수가 없어졌으므로 x가 정의되지 않았다는 메시지와 함께 NameError가 발생(리스트 사용시 유용)
x = 10
del x
#print(x)

print('\n')

print("===================================")

# 자료형
# Numbers
int_var1 = 1
print(int_var1)

print('\n')

int_var2 = -10
print(int_var2)

print('\n')

float_var1 = 15.20
print(float_var1)

print('\n')

float_var2 = 70.2-1E12
print(float_var2)

print('\n')

complex_var1 = 3.14j
print(complex_var1)

print('\n')

complex_var2 = 4.53e1-7j
print(complex_var2)

print('\n')

print("===================================")

# Strings
str = 'Hello World!'
print(str)

print(str[0])

print(str[2:5])

print(str[2:])

print(str * 2)

print(str + "TEST")

print('\n')

print("===================================")


# List
list = [ 'abcd', 786, 2.23, 'john', 70.2 ]
tinylist = [123, 'john']
print(list)

print(list[0])

print(list[1:3])

print(list[2:])

print(tinylist *2)

print(list + tinylist)

print('\n')

print("===================================")

# Tuples
tuple = ( 'abcd', 786, 2.23, 'john', 70.2 )
tinytuple = (123, 'john')
print(tuple)

print(tuple[0])

print(tuple[1:3])

print(tuple[2:])

print(tinytuple *2)

print(tuple + tinytuple)

print('\n')

print("===================================")

# Dictionary
dict = { }
dict['one'] = "This is one"
dict[2]     = "This is two"
tinydict = {'name':'john', 'code':6734, 'dept':'sales'}
print(dict['one'])

print(dict[2])

print(tinydict)

print(tinydict.keys())

print(tinydict.values())

print('\n')

print("===================================")

# 함수
# 함수 이름은 add이고 입력으로 2개의 값을 받음
# 결과는 2개의 입력을 더한 값임
def add(a, b):
    return a + b
a = 3
b = 4
c = add(a,b)
print(c)

print('\n')

print("===================================")

# 모듈
# 새 노트북 창을 열어 mo1.py 모듈 만들기
# File-Download as-Python(mo1.py) 파이썬 파일로 저장한다.
def add(a, b):
    return a + b
def sub(a, b):
    return a - b

# 모듈 불러오기
import mo1
print(mo1.add(3, 4))
print(mo1.sub(4, 2))

from mo1 import add
print(add(3, 4))
