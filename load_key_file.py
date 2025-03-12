import os
import openai
import json

def __load_file(filepath):
    with open(filepath, 'r') as file:
        return file.readline().strip()

def load_key_openai(path):
    # API 키 로드 및 환경변수 설정
    openai.api_key = __load_file(path)
    os.environ['OPENAI_API_KEY'] = openai.api_key
    
    
def load_key_naver(path):
   
    # 파일 읽기
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
 
    # API 키 및 환경변수 설정
    # {"c_id" :“your id", "c_key" : “your key"}
    os.environ['NAVER_API_ID'] = data['c_id']
    os.environ['NAVER_API_KEY'] = data['c_key']
