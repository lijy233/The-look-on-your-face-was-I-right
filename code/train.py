import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
from data_utils import MyDataset
from model import MultiModalModel

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type == 'cuda':
        print('GPU 可用！')
    else:
        print('将在 CPU 上训练...')

    print("初始化数据转换...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print("数据转换初始化完成.")

    print("加载数据集...")
    dataset = MyDataset('../emo-visual-data/data.json', '../emo-visual-data/emo/', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True)
    print("数据集加载完成.")

    print("初始化模型...")
    model = MultiModalModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print("模型初始化完成.")

    print("开始训练...")
    num_epochs = 10
    total_iterations = num_epochs * len(dataloader)
    loop = tqdm(total=total_iterations, position=0, leave=False)
    for epoch in range(num_epochs):
        for images in dataloader:
            images = images.to(device)
            optimizer.zero_grad()
            image_features = model(images)
            optimizer.step()
            loop.update(1)
    print("训练完成.")

    print("保存模型权重...")
    torch.save(model.state_dict(), '../model/model_weights.pth')
    print("模型权重保存完成.")

if __name__ == "__main__":
    main()
