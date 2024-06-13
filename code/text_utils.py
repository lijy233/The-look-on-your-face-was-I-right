import torch
from transformers import BertTokenizer, BertModel
from huaweicloudsdkcore.auth.credentials import BasicCredentials
from huaweicloudsdkocr.v1.region.ocr_region import OcrRegion
from huaweicloudsdkocr.v1 import *
import base64
from tqdm import tqdm
import os
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# 华为云 OCR 函数
def detect_text_in_image(image_path, ak, sk):
    credentials = BasicCredentials(ak, sk)
    client = OcrClient.new_builder() \
        .with_credentials(credentials) \
        .with_region(OcrRegion.value_of("cn-north-4")) \
        .build()

    try:
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

        request = RecognizeWebImageRequest()
        request.body = WebImageRequestBody(image=image_base64)
        response = client.recognize_web_image(request)
        return response.result.words_block_list
    except exceptions.ClientRequestException as e:
        print(e.status_code)
        print(e.request_id)
        print(e.error_code)
        print(e.error_msg)
        return None

# 加载和预处理文本的函数
def encode_text(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True).to(device)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()