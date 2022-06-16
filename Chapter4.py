# CSV 파일
# CSV 파일 읽기 명령 : pd_read_csv('파일경로/이름')
# 판다스 리이브러리 불러오기
import pandas as pd

# 파일경로를 찾고 변수 file_path 에 저장
file_path = 'C:/Users/USER/Desktop/아카라 코리아/파이썬/bicycle.csv'

# read csv() 함수로 데이터프레임 변환
df1 = pd.read_csv(file_path, engine='python', encoding='cp949')
df1


# EXCEL 파일
# Excel 파일 읽기 명령 : pd_read_excel('파일경로/이름')
import pandas as pd

file_path = 'C:/Users/USER/Desktop/아카라 코리아/파이썬/bicycle.xlsx'
df2 = pd.read_excel(file_path)#, encoding = 'CP949')

# 데이터프레임 출력
df2


# JSON 파일 읽기 명령 : pd.read_json('파일경로/이름')
import pandas as pd
file_path = 'C:/data/read.json'
df3 = pd.read_json(file_path)

# 데이터프레임 출력
df3


# CSV 파일 저장 명령 : df.to_csv('파일경로/이름')
# 데이터 불러오기
import pandas as pd
file_path = 'C:/data/bicycle/csv'
df4 = pd.read_csv(file_path, encoding = 'CP949')

# to_csv 함수를 사용 csv 파일로 내보내기
# data 폴더에 sample_data.csv 저장
df4.to_csv('C:/data/sample_data.csv')

# 저장 파일 확인
file_path = 'C:/data/sample_data.csv'
df5 = pd.read_csv(file_path)
print(df5)


# Excel 파일 저장 명령 : to_excel('파일경로/이름')
# 데이터 불러오기
import pandas as pd
file_path = 'C:/data/bicycle.xlsx'
df6 = pd.read_excel(file_path, encoding = 'CP949')

# to_excel 메소드를 사용하여 excel 파일로 내보내기
# 파일명은 sample_data1.xlsx로 저장
df6.to_excel('C:/data/sample_data1.xlsx')


# JSON 파일 읽기 명령 : to_json('파일경로/이름')
# 데이터 불러오기
import pandas as pd
file_path = 'C:/data/read.json'
df7 = pd.read_json(file_path)
df7

# to_json() 메소드를 사용하여 JSON 파일로 내보내기
# 파일명은 sample_data_json.json로 저장
df7.to_json ('C:/data/sample_data_json.json')


# df.to_pickle("df.pkl")
# df = pd.read_pickle("df.pkl")
import pickle
import pandas as pd

temp = pd.DataFrame({'a':[1], 'b':[2]})

# 데이터 저장
temp.to_pickle('filename.pkl')

# 데이터 로드
data = pd.read_pickle('filename.pkl')


# 파이썬으로 REST API를 요청
import pandas
from bs4 import BeautifulSoup
import requests
from openpyxl.workbook import Workbook
apikey = " 개인별로 발급받은 API 키 입력 "

# 약국정보서비스 API
api = ""

# 약국 정보 리스트
list_drugs = ["병원명", "종별코드명", "시도명", "주소", "전화번호"]
i = 0
for list_drug in list_drugs:
    url = api.format(list_drugs=list_drug, key=apikey)
    req = requests.get(url)
    re = req.text
    soup = BeautifulSoup(re, 'html.parser')

	# 병원명
    yadmnm = soup.find_all('yadmnm')
	# 종별코드명
    sggucdnm = soup.find_all('sggucdnm')
	# 시도명
    sidocdnm = soup.find_all('sidocdnm')
	# 주소
    addr = soup.find_all('addr')
	# 전화번호
    telno = soup.find_all('telno')

print("병원명:", yadmnm)
print("종별코드명:", sggucdnm)
print("시도명:", sggucdnm)
print("주소:", addr)
print("전화번호:", telno)


import pandas as pd
import requests
url_temp = "https://restcountries.eu/rest/v1/name/"
country = "South Korea"
url = url_temp + country
r = requests.get(url)
print(r.text)

# 데이터프레임으로 JSON 확인
df = pd.read_json(r.text)
df


from bs4 import BeautifulSoup as bs
from pprint import pprint
import requests

html = requests.get('https://search.naver.com/serch.naver?query=날씨')
print(html.text)


from bs4 import BeautifulSoup as bs
from pprint import pprint
import requests

html = requests.get('https://search.naver.com/search.naver?query=날씨')
soup = bs(html.text, 'html.parser')

# 미세먼지 관련 블록 추출
dustdata_one = soup.find('div', {'class':'detail_box'})
print(dustdata_one)


from bs4 import BeautifulSoup as bs
from pprint import pprint
import requests

html = requests.get('https://search.naver.com/search.naver?query=날씨')
soup = bs(html.text, 'html.parser')

# 미세먼지 관련 블록 추출
dustdata_one = soup.find('div', {'class':'detail_box'})

dustdata_all = dustdata_one.findAll('dd')
pprint(dustdata_all)


from bs4 import BeautifulSoup as bs
from pprint import pprint
import requests
html = requests.get('https://search.naver.com/search.naver?query=날씨')
soup = bs(html.text, 'html.parser')

# 미세먼지 관련 블록 추출
dustdata_one = soup.find('div', {'class':'detail_box'})

dustdata_all = dustdata_one.findAll('dd')

fine_dust_code = dustdata_all[0].find('span', {'class':'num'})
fine_dust_con = dustdata_all[0].find('span', {'class':'num'}).text

print(fine_dust_code)
print(fine_dust_con)


from bs4 import BeautifulSoup as bs
from pprint import pprint
import requests

html = requests.get('https://search.naver.com/search.naver?query=날씨')
soup = bs(html.text, 'html.parser')

dustdata_one = soup.find('div', {'class':'detail_box'})
dustdata_all = dustdata_one.findAll('dd')
find_dust_con = dustdata_all[0].find('span', {'class':'num'}).text
ultra_find_dust = dustdata_all[1].find('span', {'class':'num'}).text
print(ultra_find_dust)
