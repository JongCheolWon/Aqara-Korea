from jsonschema import Draft6Validator
import pandas as pd

file_path1 = '4월.xlsx'

df1 = pd.read_excel(file_path1, sheet_name='스마트조명')#, encoding = 'CP949')
# 데이터프레임 출력
print(df1)
df1.info()
print(df1.isnull().sum())
print('\n')


df2 = pd.read_excel(file_path1, sheet_name='온습도감지')#, encoding = 'CP949')
print(df2)
df2.info()
print(df2.isnull().sum())
print('\n')


df3 = pd.read_excel(file_path1, sheet_name='조도센서')#, encoding = 'CP949')
print(df3)
df3.info()
print(df3.isnull().sum())
print('\n')


df4 = pd.read_excel(file_path1, sheet_name='스마트큐브')#, encoding = 'CP949')
print(df4)
df4.info()
print(df4.isnull().sum())
print('\n')


df5 = pd.read_excel(file_path1, sheet_name='미니스위치')#, encoding = 'CP949')
print(df5)
df5.info()
print(df5.isnull().sum())
print('\n')

df6 = pd.read_excel(file_path1, sheet_name='열림감지')#, encoding = 'CP949')
print(df6)
df6.info()
print(df6.isnull().sum())
print('\n')


df7 = pd.read_excel(file_path1, sheet_name='스마트플러그')#, encoding = 'CP949')
print(df7)
df7.info()
print(df7.isnull().sum())
print('\n')


df8 = pd.read_excel(file_path1, sheet_name='스마트블라인드')#, encoding = 'CP949')
print(df8)
df8.info()
print(df8.isnull().sum())
