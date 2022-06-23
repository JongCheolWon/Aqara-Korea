from jsonschema import Draft6Validator
import pandas as pd

file_path1 = '1월.xlsx'
file_path2 = '2월.xlsx'
file_path3 = '3월.xlsx'
file_path4 = '4월.xlsx'


print('1월')
df1 = pd.read_excel(file_path1, sheet_name='스마트조명')#, encoding = 'CP949')
# 데이터프레임 출력
print(df1)
df1.info()
print(df1.isnull().sum()) # 결측치 확인
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
print('\n')


print('2월')

df9 = pd.read_excel(file_path2, sheet_name='스마트조명')#, encoding = 'CP949')
print(df9)
df9.info()
print(df9.isnull().sum())
print('\n')


df10 = pd.read_excel(file_path2, sheet_name='온습도감지')#, encoding = 'CP949')
print(df10)
df10.info()
print(df10.isnull().sum())
print('\n')


df11 = pd.read_excel(file_path2, sheet_name='조도센서')#, encoding = 'CP949')
print(df11)
df11.info()
print(df11.isnull().sum())
print('\n')


df12 = pd.read_excel(file_path2, sheet_name='스마트큐브')#, encoding = 'CP949')
print(df12)
df12.info()
print(df12.isnull().sum())
print('\n')


df13 = pd.read_excel(file_path2, sheet_name='미니스위치')#, encoding = 'CP949')
print(df13)
df13.info()
print(df13.isnull().sum())
print('\n')

df14 = pd.read_excel(file_path2, sheet_name='열림감지')#, encoding = 'CP949')
print(df14)
df14.info()
print(df14.isnull().sum())
print('\n')


df15 = pd.read_excel(file_path2, sheet_name='스마트플러그')#, encoding = 'CP949')
print(df15)
df15.info()
print(df15.isnull().sum())
print('\n')


df16 = pd.read_excel(file_path2, sheet_name='스마트블라인드')#, encoding = 'CP949')
print(df16)
df16.info()
print(df16.isnull().sum())
print('\n')


print('3월')

df17 = pd.read_excel(file_path3, sheet_name='스마트조명')#, encoding = 'CP949')
print(df17)
df17.info()
print(df17.isnull().sum())
print('\n')


df18 = pd.read_excel(file_path3, sheet_name='온습도감지')#, encoding = 'CP949')
print(df18)
df18.info()
print(df18.isnull().sum())
print('\n')


df19 = pd.read_excel(file_path3, sheet_name='조도센서')#, encoding = 'CP949')
print(df19)
df19.info()
print(df19.isnull().sum())
print('\n')


df20 = pd.read_excel(file_path3, sheet_name='스마트큐브')#, encoding = 'CP949')
print(df20)
df20.info()
print(df20.isnull().sum())
print('\n')


df21 = pd.read_excel(file_path3, sheet_name='미니스위치')#, encoding = 'CP949')
print(df21)
df21.info()
print(df21.isnull().sum())
print('\n')

df22 = pd.read_excel(file_path3, sheet_name='열림감지')#, encoding = 'CP949')
print(df22)
df22.info()
print(df22.isnull().sum())
print('\n')


df23 = pd.read_excel(file_path3, sheet_name='스마트플러그')#, encoding = 'CP949')
print(df23)
df23.info()
print(df23.isnull().sum())
print('\n')


df24 = pd.read_excel(file_path3, sheet_name='스마트블라인드')#, encoding = 'CP949')
print(df24)
df24.info()
print(df24.isnull().sum())
print('\n')


print('4월')

df25 = pd.read_excel(file_path4, sheet_name='스마트조명')#, encoding = 'CP949')
print(df25)
df25.info()
print(df25.isnull().sum())
print('\n')


df26 = pd.read_excel(file_path4, sheet_name='온습도감지')#, encoding = 'CP949')
print(df26)
df26.info()
print(df26.isnull().sum())
print('\n')


df27 = pd.read_excel(file_path4, sheet_name='조도센서')#, encoding = 'CP949')
print(df27)
df27.info()
print(df27.isnull().sum())
print('\n')


df28 = pd.read_excel(file_path4, sheet_name='스마트큐브')#, encoding = 'CP949')
print(df28)
df28.info()
print(df28.isnull().sum())
print('\n')


df29 = pd.read_excel(file_path4, sheet_name='미니스위치')#, encoding = 'CP949')
print(df29)
df29.info()
print(df29.isnull().sum())
print('\n')

df30 = pd.read_excel(file_path4, sheet_name='열림감지')#, encoding = 'CP949')
print(df30)
df30.info()
print(df30.isnull().sum())
print('\n')


df31 = pd.read_excel(file_path4, sheet_name='스마트플러그')#, encoding = 'CP949')
print(df31)
df31.info()
print(df31.isnull().sum())
print('\n')


df32 = pd.read_excel(file_path4, sheet_name='스마트블라인드')#, encoding = 'CP949')
print(df32)
df32.info()
print(df32.isnull().sum())
