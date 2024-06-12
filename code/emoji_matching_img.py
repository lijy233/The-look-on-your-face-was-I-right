import os
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
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

if __name__ == "__main__":
    json_path = "../emo-visual-data/data.json"
    image_folder = "../emo-visual-data/emo"
    features_path = "../model/features.npy"
    filenames_path = "../model/filenames.npy"
    description_embeddings_path = "../model/description_embeddings.npy"
    ak = "EPQ2BDPM3AWRYOEBPVMJ"  # 这里替换为您的华为云 AK
    sk = "XEfYiba9yFUXLMc27aLI86MnVM9krZuLPoXcLvDT"  # 这里替换为您的华为云 SK

    print("Loading data...")
    data = load_data(json_path)
    print("Data loaded.")

    print("Loading tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', cache_dir='./cache', local_files_only=True)
    text_model = BertModel.from_pretrained('bert-base-chinese', cache_dir='./cache', local_files_only=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text_model.to(device)
    print(f"Tokenizer and model loaded. Using device: {device}")

    model = MultiModalModel().to(device)
    model.load_state_dict(torch.load('../model/model_weights.pth'))
    model.eval()

    # 加载提取的图像特征和文件名
    print("Loading pre-extracted features, filenames and description embeddings...")
    features = np.load(features_path)
    filenames = np.load(filenames_path)
    description_embeddings = np.load(description_embeddings_path)
    print("Features, filenames and description embeddings loaded.")

    input_img_path = input("Enter path to your image: ")
    
    print("Detecting text in the image...")
    words_block_list = detect_text_in_image(input_img_path, ak, sk)
    has_text = len(words_block_list) > 0 if words_block_list else False
    print("Text detection completed.")

    input_img_tensor = load_and_preprocess_image(input_img_path, device=device)
    with torch.no_grad():
        input_feature = model(input_img_tensor).flatten().cpu().numpy()  # 将结果移回CPU
    similarities = cosine_similarity([input_feature], features)
    
    if has_text:
        print("Processing detected text...")
        # 将所有检测到的文字拼接成一个字符串
        detected_text = " ".join([block.words for block in words_block_list])
        input_text_embedding = encode_text(detected_text, tokenizer, text_model, device)
        
        # 计算文本和图像的相似度
        combined_similarities = []
        for i, filename in enumerate(tqdm(filenames, desc="Calculating combined similarities")):
            text_similarity = np.dot(input_text_embedding, description_embeddings[i].T).item()
            combined_similarity = text_similarity + similarities[0][i]
            combined_similarities.append(combined_similarity)
        
        most_similar_idx = np.argmax(combined_similarities)
    else:
        most_similar_idx = np.argmax(similarities)

    most_similar_image = filenames[most_similar_idx]
    print(f"Your emoji is: {most_similar_image}")

    def display_images(img_path1, img_path2):
        img1 = Image.open(img_path1)
        img2 = Image.open(img_path2)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img1)
        axes[0].set_title("Input Image")
        axes[0].axis('off')
        axes[1].imshow(img2)
        axes[1].set_title("Your emoji, did I guess it right?")
        axes[1].axis('off')
        plt.show()

    display_images(input_img_path, os.path.join(image_folder, most_similar_image))


