
import os
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import openai
from openai import OpenAI
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
import numpy as np

# 0. load key file------------------
import load_key_file as lk
def load_all(path):# 키 경로를 입력하면 키를 환경변수에 등록 하고 모델 토크나이저 응급실 정보등을 가져옴 
  lk.load_key_openai(path+'api_key.txt')
  lk.load_key_naver(path+'map_key.txt')
  client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
  model_directory=path + "fine_tuned_bert"
  model = AutoModelForSequenceClassification.from_pretrained(model_directory)
  tokenizer = AutoTokenizer.from_pretrained(model_directory)
  hospital_info=pd.read_csv(path+'응급실정보.csv')
  return client, model, tokenizer, hospital_info
# 1-1 audio2text--------------------
import audio2text as att
def  a2t(audiopath,client):
  text = att.audio2text(audiopath,client)
  print(text) # !!!임시 테스트용 print!!!
  return text

# 1-2 text2summary------------------
import text2summary as tts
def  t2s(text,client):
  summary = tts.text2summary(text,client)
  print(summary) # !!!임시 테스트용 print!!!
  return summary

# 2. model prediction------------------
def predict(text, model, tokenizer,device='cpu'):
    # 입력 문장 토크나이징
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # 각 텐서를 GPU로 이동
    model = model.to(device)
    model.eval()
    # 모델 예측
    with torch.no_grad():
        outputs = model(**inputs)
    # 로짓을 소프트맥스로 변환하여 확률 계산
    logits = outputs.logits
    probabilities = logits.softmax(dim=1)
    # 가장 높은 확률을 가진 클래스 선택
    pred = torch.argmax(probabilities, dim=-1).item()
    return pred
# 3-1. get_distance------------------
def get_dist(start_lat, start_lng, dest_lat, dest_lng, c_id, c_key):
    url = "https://naveropenapi.apigw.ntruss.com/map-direction/v1/driving"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": c_id,
        "X-NCP-APIGW-API-KEY": c_key,
    }
    params = {
        "start": f"{start_lng},{start_lat}",  # 출발지 (경도, 위도)
        "goal": f"{dest_lng},{dest_lat}",    # 목적지 (경도, 위도)
        "option": "trafast"  # 실시간 빠른 길 옵션
    }

    # 요청하고, 답변 받아오기
    response = requests.get(url, headers=headers, params=params).json()
    if response['code'] == 0:
        dist = response['route']['trafast'][0]['summary']['distance']  # m(미터)
    elif response['code'] == 1:
        dist = 0
    else:
      dist=np.inf
    return dist
# 3-2. recommendation------------------
def recommendation(coords,data, c_id, c_key):
  lat,lng = coords
  nearest_hospital=data.loc[(abs(data['위도'] - lat) <= 0.1)&(abs(data['경도'] - lng) <= 0.1)].copy()
  nearest_hospital['거리'] = nearest_hospital.apply(lambda row: get_dist(lat, lng, row['위도'], row['경도'], c_id, c_key), axis=1)
  return nearest_hospital.sort_values(by='거리')[:3]

