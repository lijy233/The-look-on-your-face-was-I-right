import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
from data_utils import MyDataset
from model import MultiModalModel

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 检查是否有GPU可用
    
    if device.type == 'cuda':
        print('GPU is available!')
    else:
        print('Training on CPU...')

    # 定义数据转换
    print("Initializing data transformations...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print("Data transformations initialized.")

    # 加载数据集
    print("Loading dataset...")
    dataset = MyDataset('../emo-visual-data/data.json', '../emo-visual-data/emo/', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True)
    print("Dataset loaded.")

    # 初始化模型并将其移动到GPU
    print("Initializing model...")
    model = MultiModalModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print("Model initialized.")

    # 训练模型
    print("Starting training...")
    num_epochs = 10
    total_iterations = num_epochs * len(dataloader)
    loop = tqdm(total=total_iterations, position=0, leave=False)
    for epoch in range(num_epochs):
        for images in dataloader:
            images = images.to(device)  # 将数据移动到GPU
            optimizer.zero_grad()
            image_features = model(images)
            optimizer.step()
            loop.update(1)  # 更新进度条

    print("Training completed.")

    # 保存模型权重
    print("Saving model weights...")
    torch.save(model.state_dict(), '../model/model_weights.pth')
    print("Model weights saved.")
