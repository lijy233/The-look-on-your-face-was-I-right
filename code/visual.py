import os
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
from data_utils import load_data, load_and_preprocess_image
from model import MultiModalModel
import base64
from huaweicloudsdkcore.auth.credentials import BasicCredentials
from huaweicloudsdkocr.v1.region.ocr_region import OcrRegion
from huaweicloudsdkcore.exceptions import exceptions
from huaweicloudsdkocr.v1 import *


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
        request.body = WebImageRequestBody(
            image=image_base64
        )
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


# 加载模型和数据
def load_model_and_data():
    json_path = "../emo-visual-data/data.json"
    image_folder = "../emo-visual-data/emo"
    features_path = "../model/features.npy"
    filenames_path = "../model/filenames.npy"
    description_embeddings_path = "../model/description_embeddings.npy"
    ak = "EPQ2BDPM3AWRYOEBPVMJ"  # 这里替换为您的华为云 AK
    sk = "XEfYiba9yFUXLMc27aLI86MnVM9krZuLPoXcLvDT"  # 这里替换为您的华为云 SK

    data = load_data(json_path)

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', cache_dir='./cache', local_files_only=True)
    text_model = BertModel.from_pretrained('bert-base-chinese', cache_dir='./cache', local_files_only=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text_model.to(device)

    model = MultiModalModel().to(device)
    model.load_state_dict(torch.load('../model/model_weights.pth'))
    model.eval()

    features = np.load(features_path)
    filenames = np.load(filenames_path)
    description_embeddings = np.load(description_embeddings_path)

    return data, tokenizer, text_model, device, model, description_embeddings, ak, sk, image_folder, features, filenames


# 主应用程序
def main():
    st.title('你的表情我猜对了吗？')
    st.write('这是一个通过文本或图片匹配相应表情包的项目。')

    choice = st.selectbox('请选择一种方式匹配表情包:', ('通过文字匹配', '通过图片匹配'))

    if choice == '通过文字匹配':
        st.subheader('通过文字匹配')
        input_text = st.text_input('请输入你的文本:')
        if input_text:
            data, tokenizer, text_model, device, model, description_embeddings, ak, sk, image_folder, _, _ = load_model_and_data()
            input_embedding = encode_text(input_text, tokenizer, text_model, device)

            best_match = None
            highest_similarity = float('-inf')

            for i, item in enumerate(tqdm(data, desc="Processing descriptions")):
                similarity = np.dot(input_embedding, description_embeddings[i].T).item()

                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = item

            if best_match:
                image_path = os.path.join(image_folder, best_match['filename'])
                img = Image.open(image_path)
                st.image(img, caption='匹配的表情包:', use_column_width=True)
            else:
                st.write("没有找到匹配的描述。")

    elif choice == '通过图片匹配':
        st.subheader('通过图片匹配')
        uploaded_file = st.file_uploader("上传一张图片:", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            with open("temp.jpg", "wb") as f:
                f.write(uploaded_file.read())
            input_img_path = "temp.jpg"

            data, tokenizer, text_model, device, model, description_embeddings, ak, sk, image_folder, features, filenames = load_model_and_data()
            words_block_list = detect_text_in_image(input_img_path, ak, sk)
            has_text = len(words_block_list) > 0 if words_block_list else False

            input_img_tensor = load_and_preprocess_image(input_img_path, device=device)
            with torch.no_grad():
                input_feature = model(input_img_tensor).flatten().cpu().numpy()

            similarities = cosine_similarity([input_feature], features)

            if has_text:
                detected_text = " ".join([block.words for block in words_block_list])
                input_text_embedding = encode_text(detected_text, tokenizer, text_model, device)

                combined_similarities = []
                for i, filename in enumerate(tqdm(filenames, desc="Calculating combined similarities")):
                    text_similarity = np.dot(input_text_embedding, description_embeddings[i].T).item()
                    combined_similarity = text_similarity + similarities[0][i]
                    combined_similarities.append(combined_similarity)

                most_similar_idx = np.argmax(combined_similarities)
            else:
                most_similar_idx = np.argmax(similarities)

            most_similar_image = filenames[most_similar_idx]

            input_img = Image.open(input_img_path)
            matched_img_path = os.path.join(image_folder, most_similar_image)
            matched_img = Image.open(matched_img_path)

            st.image(input_img, caption='上传的图片:', use_column_width=True)
            st.image(matched_img, caption='匹配的表情包:', use_column_width=True)

    else:
        st.write("无效的选择。")


if __name__ == "__main__":
    main()
