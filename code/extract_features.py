import os
import numpy as np
import torch
from tqdm import tqdm
from data_utils import load_data, load_and_preprocess_image
from model import MultiModalModel

def extract_and_save_features(json_path, image_folder, model_weights_path, features_path, filenames_path):
    # 加载数据和模型
    data = load_data(json_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MultiModalModel().to(device)
    model.load_state_dict(torch.load(model_weights_path))
    model.eval()

    # 提取表情包图片特征
    print("Extracting image features...")
    features = []
    filenames = []

    for item in tqdm(data, desc="Processing emoji images"):
        img_path = os.path.join(image_folder, item['filename'])
        img_tensor = load_and_preprocess_image(img_path, device=device)
        with torch.no_grad():
            feature = model(img_tensor).flatten().cpu().numpy()  # 提取特征并将结果移回CPU
        features.append(feature)
        filenames.append(item['filename'])

    features = np.array(features)
    
    # 确保保存特征和文件名的目录存在
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    os.makedirs(os.path.dirname(filenames_path), exist_ok=True)
    
    # 保存特征和文件名
    np.save(features_path, features)
    np.save(filenames_path, filenames)
    print("Features and filenames saved.")

if __name__ == "__main__":
    json_path = "../emo-visual-data/data.json"
    image_folder = "../emo-visual-data/emo"
    model_weights_path = '../model/model_weights.pth'
    features_path = '../model/emo_features.npy'
    filenames_path = '../model/emo_filenames.npy'

    extract_and_save_features(json_path, image_folder, model_weights_path, features_path, filenames_path)

