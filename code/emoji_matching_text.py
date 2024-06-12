# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import matplotlib.pyplot as plt

# 加载和预处理文本的函数
def encode_text(text_list, tokenizer, model, device):
    inputs = tokenizer(text_list, return_tensors='pt', truncation=True, max_length=512, padding=True).to(device)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()

def find_best_match(input_text, filenames, description_embeddings):
    description_embeddings = description_embeddings.reshape(description_embeddings.shape[0], -1)
    similarities = cosine_similarity(input_text, description_embeddings)
    best_match_idx = np.argmax(similarities)
    return filenames[best_match_idx]

if __name__ == "__main__":
    json_path = "../emo-visual-data/data.json"
    image_folder = "../emo-visual-data/emo"
    filenames_path = "../model/filenames.npy"
    description_embeddings_path = "../model/description_embeddings.npy"

    print("Loading tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', cache_dir='./cache', local_files_only=True)
    text_model = BertModel.from_pretrained('bert-base-chinese', cache_dir='./cache', local_files_only=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text_model.to(device)
    print(f"Tokenizer and model loaded. Using device: {device}")

    print("Loading data...")
    filenames = np.load(filenames_path)
    description_embeddings = np.load(description_embeddings_path)
    print("Data loaded.")

    input_text = input("Enter your text: ")
    input_embedding = encode_text([input_text], tokenizer, text_model, device)

    best_match_filename = find_best_match(input_embedding, filenames, description_embeddings)
    print(f"Your emoji is: {best_match_filename}")

    def display_image(image_path):
        img = Image.open(image_path)
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    display_image(os.path.join(image_folder, best_match_filename))
