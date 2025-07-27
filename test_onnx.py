import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from test import predict
from myDataset import MyDataset
from myModules import compute_class_prototypes
import pandas as pd
import onnxruntime

if __name__ == '__main__':
    # 检查是否有可用的 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据预处理
    transform = transforms.Compose([
                    transforms.Resize((153, 125)),
                    transforms.ToTensor()
                ])

    session = onnxruntime.InferenceSession('food.onnx')

    # 创建 Dataset
    root_dir = r'爆炒肉片new'
    file_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir)
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
    dataset = MyDataset(file_paths, transform=transform)

    # 创建 DataLoader
    batch_size = len(dataset)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    class_prototypes = compute_class_prototypes(session, dataloader, device)
    
    df = pd.read_csv('测试集列表.csv', header=None)
    test_set : list[list[str]] = df.values.tolist()
    datasetTest = MyDataset([x for row in test_set for x in row], transform=transform)

    df_pred = pd.DataFrame()
    embeddings = []
    with torch.no_grad():
        for t, name_label, prefix_label, star_num in datasetTest:
            name = datasetTest.label_prefix_dict[prefix_label] + datasetTest.label_name_dict[name_label] + '★' * star_num
            print(f'应该是：{name}')
            # 提取特征向量
            embedding : torch.Tensor
            embedding, distance, pred_name_label, pred_prefix_label, pred_star_num = predict(session, class_prototypes, t, device)
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

