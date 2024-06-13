import cv2
import numpy as np
import torch
from transformers import BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import os
from PIL import Image
from data_utils import load_and_preprocess_image
from text_utils import detect_text_in_image, encode_text
from model import MultiModalModel
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

def capture_image_from_camera(face_cascade, text_model, model, device, tokenizer, description_embeddings, filenames, ak, sk, image_folder, features):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 检测面部
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # 绘制面部边框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 保存检测到的面部图像
            cap.release()
            cv2.destroyAllWindows()

            # 保存图像
            input_img_path = "captured_image.jpg"
            cv2.imwrite(input_img_path, frame)

            print("正在检测图像中的文字...")
            words_block_list = detect_text_in_image(input_img_path, ak, sk)
            has_text = len(words_block_list) > 0 if words_block_list else False
            print("文字检测完成.")

            input_img_tensor = load_and_preprocess_image(input_img_path, device=device)
            with torch.no_grad():
                input_feature = model(input_img_tensor).flatten().cpu().numpy()

            similarities = cosine_similarity([input_feature], features)

            if has_text:
                print("正在处理检测到的文字...")
                # 将所有检测到的文字拼接成一个字符串
                detected_text = " ".join([block.words for block in words_block_list])
                input_text_embedding = encode_text(detected_text, tokenizer, text_model, device)

                # 计算文本和图像的相似度
                combined_similarities = []
                for i, filename in enumerate(tqdm(filenames, desc="正在计算组合相似度")):
                    text_similarity = np.dot(input_text_embedding, description_embeddings[i].T).item()
                    combined_similarity = text_similarity + similarities[0][i]
                    combined_similarities.append(combined_similarity)

                most_similar_idx = np.argmax(combined_similarities)
            else:
                most_similar_idx = np.argmax(similarities)

            most_similar_image = filenames[most_similar_idx]
            print(f"您的表情包是: {most_similar_image}")

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
            return

        # 使用matplotlib显示图像
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title('面部检测')
        plt.axis('off')
        plt.show()

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return None
