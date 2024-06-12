import os
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from data_utils import load_data, load_and_preprocess_image
from model import MultiModalModel

# 加载和预处理文本的函数
def encode_text(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True).to(device)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()

json_path = "../emo-visual-data/data.json"
image_folder = "../emo-visual-data/emo"
features_path = "../model/features.npy"
filenames_path = "../model/filenames.npy"
description_embeddings_path = "../model/description_embeddings.npy"

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

features = []
filenames = []
description_embeddings = []

print("Extracting image features and description embeddings...")
for item in tqdm(data, desc="Processing emoji images"):
    image_path = os.path.join(image_folder, item['filename'])
    input_img_tensor = load_and_preprocess_image(image_path, device=device)
    
    with torch.no_grad():
        feature = model(input_img_tensor).flatten().cpu().numpy()
    features.append(feature)
    filenames.append(item['filename'])
    
    description_embedding = encode_text(item['content'], tokenizer, text_model, device)
    description_embeddings.append(description_embedding)

features = np.array(features)
description_embeddings = np.array(description_embeddings)

os.makedirs(os.path.dirname(features_path), exist_ok=True)
np.save(features_path, features)
np.save(filenames_path, filenames)
np.save(description_embeddings_path, description_embeddings)

print("Feature extraction completed.")
