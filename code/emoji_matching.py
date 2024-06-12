import os
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import matplotlib.pyplot as plt
from PIL import Image
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

# 寻找最匹配的表情符号函数
def find_best_match(input_text, data, tokenizer, model, device):
    print("Encoding input text...")
    input_embedding = encode_text(input_text, tokenizer, model, device)
    
    best_match = None
    highest_similarity = float('-inf')
    
    print("Finding best match...")
    for item in tqdm(data, desc="Processing descriptions"):
        description = item['content']
        description_embedding = encode_text(description, tokenizer, model, device)
        similarity = np.dot(input_embedding, description_embedding.T).item()
        
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = item
    
    return best_match

# 提取所有表情包图片的特征
def extract_image_features(emo_folder, data, model, device):
    features = []
    filenames = []

    for item in tqdm(data, desc="Processing emoji images"):
        img_path = os.path.join(emo_folder, item['filename'])
        img_tensor = load_and_preprocess_image(img_path, device=device)
        with torch.no_grad():
            feature = model(img_tensor).flatten().cpu().numpy()  # 提取特征并将结果移回CPU
        features.append(feature)
        filenames.append(item['filename'])

    features = np.array(features)
    return features, filenames

if __name__ == "__main__":
    json_path = "../emo-visual-data/data.json"
    image_folder = "../emo-visual-data/emo"
    ak = "EPQ2BDPM3AWRYOEBPVMJ"  # 这里替换为您的华为云 AK
    sk = "XEfYiba9yFUXLMc27aLI86MnVM9krZuLPoXcLvDT"  # 这里替换为您的华为云 SK

    print("Loading data...")
    data = load_data(json_path)
    print("Data loaded.")

    print("Loading tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', cache_dir='./cache')
    text_model = BertModel.from_pretrained('bert-base-chinese', cache_dir='./cache')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text_model.to(device)
    print(f"Tokenizer and model loaded. Using device: {device}")

    model = MultiModalModel().to(device)
    model.load_state_dict(torch.load('../model/model_weights.pth'))
    model.eval()

    input_choice = input("Enter your choice (1 for text, 2 for image): ")
    if input_choice == '1':
        input_text = input("Enter your text: ")
        best_match = find_best_match(input_text, data, tokenizer, text_model, device)
        if best_match:
            image_path = os.path.join(image_folder, best_match['filename'])
            img = Image.open(image_path)
            plt.imshow(img)
            plt.axis('off')
            plt.show()
        else:
            print("No matching description found.")
    elif input_choice == '2':
        input_img_path = input("Enter path to your image: ")
        
        print("Detecting text in the image...")
        words_block_list = detect_text_in_image(input_img_path, ak, sk)
        has_text = len(words_block_list) > 0 if words_block_list else False
        print("Text detection completed.")
        
        print("Extracting image features...")
        features, filenames = extract_image_features(image_folder, data, model, device)
        print("Image features extracted.")

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
                description = next(item['content'] for item in data if item['filename'] == filename)
                description_embedding = encode_text(description, tokenizer, text_model, device)
                text_similarity = np.dot(input_text_embedding, description_embedding.T).item()
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
    else:
        print("Invalid choice.")
