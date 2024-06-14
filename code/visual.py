import os
import numpy as np
import torch
import sys
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
import cv2
import matplotlib
matplotlib.use('TkAgg')
from model import MultiModalModel
from data_utils import load_data, load_and_preprocess_image
from text_utils import detect_text_in_image, encode_text
from camera_utils import capture_image_from_camera
from emotion_analysis import analyze_sentiment
from BLEU import calculate_bleu_score

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
    model.load_state_dict(torch.load('../model/model_weights.pth', map_location=device))
    model.eval()

    features = np.load(features_path)
    filenames = np.load(filenames_path)
    description_embeddings = np.load(description_embeddings_path)

    return data, tokenizer, text_model, device, model, description_embeddings, ak, sk, image_folder, features, filenames

# 主应用程序
def main():
    # 添加logo图
    logo_path = "../logo/bee-705412_1280.webp"
    st.sidebar.image(logo_path, width=200)
    st.title('你的表情 ，我猜对了吗？')
    st.write('欢迎体验我们的表情包匹配项目！在这里，你可以通过输入一段文字来描述你此刻的心情，我们将为你猜测并匹配一张符合你心情的表情包图片。当然，你也可以直接上传一张表情包图片，我们会生成一张与之相似的表情包图片。快来试试吧，让我们帮你找到最贴合心情的表情包！')

    choice = st.sidebar.selectbox('请选择一种方式匹配表情包:', ('通过文字匹配', '通过图片匹配', '通过摄像头匹配'))
    args = sys.argv
    arguments = args[1] if len(args) > 1 else True

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
                normalized_similarity = (similarity + 1) / 2  # 将点积映射到 [0, 1] 范围
                if normalized_similarity > highest_similarity:
                    highest_similarity = normalized_similarity
                    best_match = item
            if arguments:#等于True输出相似度和BLUE，
                st.write("相似度：",highest_similarity)

            if best_match:
                image_path = os.path.join(image_folder, best_match['filename'])
                img = Image.open(image_path)
                st.image(img, caption='匹配的表情包', use_column_width=True)
                # 情感分析
                sentiment = analyze_sentiment(input_text)
                st.write(f"情感分析结果: {sentiment}")

                if sentiment == "Positive":
                    st.write("你现在的心情看起来很积极！希望这张表情包能传递你的快乐。")
                else:
                    st.write("你现在的心情似乎有些消极。这张表情包也许能帮你表达情绪。")
                if arguments:  # 等于True输出相似度和BLUE，
                    reference_description = best_match['content']
                    bleu_score = calculate_bleu_score(input_text, reference_description)
                    print(f"BLEU Score: {bleu_score}")
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
            similarities = (similarities + 1) / 2  # 将余弦相似度映射到 [0, 1] 范围

            if has_text:
                detected_text = "".join([block.words for block in words_block_list])
                input_text_embedding = encode_text(detected_text, tokenizer, text_model, device)

                combined_similarities = []
                for i, filename in enumerate(tqdm(filenames, desc="Calculating combined similarities")):
                    text_similarity = cosine_similarity(input_text_embedding, description_embeddings[i]).flatten()[0]
                    text_similarity = (text_similarity + 1) / 2
                    combined_similarity = 0.9 * text_similarity + 0.1 * similarities[0][i]
                    combined_similarities.append(combined_similarity)

                most_similar_idx = np.argmax(combined_similarities)
                if arguments:
                    # print("相似度：", combined_similarities[most_similar_idx])
                    st.write("相似度：", combined_similarities[most_similar_idx])
            else:
                most_similar_idx = np.argmax(similarities)
                if arguments:
                    similarity_score = similarities[0][most_similar_idx]
                    st.write("相似度：",similarity_score)
            most_similar_image = filenames[most_similar_idx]


            input_img = Image.open(input_img_path)
            matched_img_path = os.path.join(image_folder, most_similar_image)
            matched_img = Image.open(matched_img_path)

            st.image(input_img, caption='上传的图片', use_column_width=True)
            st.image(matched_img, caption='匹配的表情包', use_column_width=True)


    elif choice == '通过摄像头匹配':

        st.subheader('通过摄像头匹配')
        start_camera = st.button("请点击来允许摄像头拍摄并匹配")
        if start_camera:

             # 加载模型和数据
            data, tokenizer, text_model, device, model, description_embeddings, ak, sk, image_folder, features, filenames = load_model_and_data()
            # 加载OpenCV的Haar级联分类器
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
             # 捕捉摄像头图像并匹配表情包
            capture_image_from_camera(face_cascade, text_model, model, device, tokenizer, description_embeddings, filenames,
                                  ak, sk, image_folder, features)

    else:
        st.write("无效的选择。")
        
