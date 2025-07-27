import os
import onnxruntime
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from myDataset import MyDataset
from myModules import PrototypicalNetwork, compute_class_prototypes
import pandas as pd

def predict(model : nn.Module|onnxruntime.InferenceSession, class_prototypes, image, device):
    """
    使用 PrototypicalNetwork 和类原型对单张图像进行分类
    Args:
        model: 训练好的 PrototypicalNetwork 模型
        class_prototypes: 类原型字典
        image: 待预测的图像（形状为 [3, 125, 153]）
        device: 设备（'cuda' 或 'cpu'）
    Returns:
        embedding: 特征向量
        distance: 最近的欧氏距离
        name_label: 名称标签
        prefix_label: 前缀标签
        star_num: 星星数
    """
    if isinstance(model, nn.Module):
        model.eval()  # 将模型设置为评估模式

    # 将图像包装为 batch（形状从 [3, 125, 153] 变为 [1, 3, 125, 153]）
    image = image.unsqueeze(0).to(device)

    # 提取特征向量
    with torch.no_grad():
        if isinstance(model, nn.Module):
            embedding, prefix_logist, star_logist = model(image)  # [1, 64]
        elif isinstance(model, onnxruntime.InferenceSession):
            input_names = [i.name for i in model.get_inputs()]
            embedding, prefix_logist, star_logist = model.run(None, {input_names[0] : image.detach().cpu().numpy()})  # [1, 64]
            embedding = torch.from_numpy(embedding).to(device)
            prefix_logist = torch.from_numpy(prefix_logist).to(device)
            star_logist = torch.from_numpy(star_logist).to(device)
        else:
            raise Exception()

    # 计算与每个类原型之间的距离（欧氏距离）
    distances = {}
    for label, prototype in class_prototypes.items():
        prototype = prototype.to(device)
        distance = torch.norm(embedding - prototype, p=2)  # 欧氏距离
        distances[label] = distance.item()

    # 找到距离最小的类别
    name_label = min(distances, key=distances.get)

    # 计算prefix_label
    softmax = nn.Softmax(1)
    prefix_label = torch.argmax(softmax(prefix_logist)).item()
    star_num = torch.argmax(softmax(star_logist)).item()

    return embedding, distance, name_label, prefix_label, star_num


if __name__ == '__main__':
    # 检查是否有可用的 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据预处理
    transform = transforms.Compose([
                    transforms.Resize((153, 125)),
                    transforms.ToTensor()
                ])

    model = PrototypicalNetwork()
    model.load_state_dict(torch.load("model.pth", weights_only=True))

    # 创建 Dataset
    root_dir = r'爆炒肉片new'
    file_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir)
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
    dataset = MyDataset(file_paths, transform=transform)

    # 创建 DataLoader
    batch_size = len(dataset)

    model.to(device)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    class_prototypes = compute_class_prototypes(model, dataloader, device)
    
    df = pd.read_csv('测试集列表.csv', header=None)
    test_set : list[list[str]] = df.values.tolist()
    datasetTest = MyDataset([x for row in test_set for x in row], transform=transform)
    model.eval()

    df_pred = pd.DataFrame()
    embeddings = []
    with torch.no_grad():
        for t, name_label, prefix_label, star_num in datasetTest:
            name = datasetTest.label_prefix_dict[prefix_label] + datasetTest.label_name_dict[name_label] + '★' * star_num
            print(f'应该是：{name}')
            # 提取特征向量
            embedding : torch.Tensor
            embedding, distance, pred_name_label, pred_prefix_label, pred_star_num = predict(model, class_prototypes, t, device)
            embeddings.append(embedding.squeeze().detach().cpu().numpy())

            name_predict = dataset.label_prefix_dict[pred_prefix_label] + dataset.label_name_dict[pred_name_label] + '★' * pred_star_num
            print(f'预测是：{name_predict}, {distance}')

            pred_dict = {}
            pred_dict['标注名称'] = name
            pred_dict['预测名称'] = name_predict
            pred_dict['预测距离'] = f'{distance:.4f}'
            df_pred = pd.concat([df_pred, pd.DataFrame.from_dict(pred_dict, orient='index').T], ignore_index=True)

    import numpy
    # 保存为本地的 npy 文件
    numpy.save('测试集特征.npy', embeddings)

    df = pd.DataFrame()
    df['图像路径'] = [tp[0] for tp in datasetTest.data]

    df = pd.concat([df, df_pred], axis=1)
    df.to_csv('测试集预测结果.csv', index=False)

