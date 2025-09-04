import torch
import torch.nn as nn
import onnxruntime

class PrototypicalNetwork(nn.Module):
    def __init__(self):
        super(PrototypicalNetwork, self).__init__()
        # 简单的卷积网络
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # [batch_size, 32, 62, 62]
        )
        self.star_fc = nn.Sequential(
            nn.AvgPool2d(kernel_size=(10, 6), stride=(10, 6)),  # [32, 10, 60] -> [32, 1, 10]
            nn.Flatten(),
            nn.Linear(32 * 10, 6)
        )
        # self.star_lambda = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.is_food_detector = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),  # [32, 16, 16] -> [32, 8, 8]
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 1),
            nn.Sigmoid()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # self.fc1 = nn.Linear(128 * 15 * 15, 256)  # 需要根据输入尺寸计算
        # self.fc2 = nn.Linear(256, 64)  # 输出特征维度
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3)  # [batch_size, 128, 5, 5]
        self.fc2 = nn.Linear(128 * 5 * 5, 64)
        self.prefix_fc = nn.Linear(64, 3)
        # self.prefix_lambda = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, x:torch.Tensor):
        # 输入 x: [batch_size, 3, 125, 125]
        x = self.conv1(x)  # [batch_size, 32, 62, 62]
        # 截取底部区域给star层
        bottom_crop = x[:, :, -10:, 1:61]   # [32, 10, 60]
        star = self.star_fc(bottom_crop)
        top_left_crop = x[:, :, :16, :16]   # [32, 16, 16]
        is_food_prob = self.is_food_detector(top_left_crop).squeeze(1)
        is_food = is_food_prob > 0.5
        x = self.conv2(x)  # [batch_size, 64, 31, 31]
        x = self.conv3(x)  # [batch_size, 128, 15, 15]
        # x = x.view(x.size(0), -1)  # 展平 [batch_size, 128 * 19 * 15]
        # x = nn.functional.relu(self.fc1(x))  # [batch_size, 256]
        # x = self.fc2(x)  # [batch_size, 64]
        # x = nn.functional.pad(x, (0, 0, 0, 1)) # 下方填充1行
        x = self.pool2(x)  # [batch_size, 128, 5, 5]
        x = x.view(x.size(0), -1)
        x = self.fc2(x)
        prefix = self.prefix_fc(x)
        # x = torch.cat([x, self.prefix_lambda * prefix, self.star_lambda * star], dim=1)
        # x = x.detach()    # 梯度解耦不太适用
        food_only_prefix = torch.where(
            is_food.unsqueeze(1),   # where自动广播到符合prefix的Nx3
            prefix,
            torch.zeros_like(prefix)
        )   # 如果不是is_food，则返回全零

        return x, food_only_prefix, star, is_food_prob


def compute_class_prototypes(model : nn.Module|onnxruntime.InferenceSession, dataloader, device):
    """
    计算每个类别的原型（即该类所有样本特征向量的均值）
    Args:
        model: 训练好的 PrototypicalNetwork 模型
        dataloader: 包含训练数据的 DataLoader
        device: 设备（'cuda' 或 'cpu'）
    Returns:
        class_prototypes: 一个字典，键是类别标签，值是该类的原型（特征向量）
    """
    if isinstance(model, nn.Module):
        model.eval()  # 将模型设置为评估模式
    class_prototypes : dict[int, torch.Tensor] = {}  # 存储每个类别的原型
    class_counts = {}      # 存储每个类别的样本数量

    with torch.no_grad():  # 禁用梯度计算
        for images, name_labels, _, _, _ in dataloader:
            images : torch.Tensor
            images, name_labels = images.to(device), name_labels.to(device)

            # 提取特征向量
            if isinstance(model, nn.Module):
                embeddings, _, _, _ = model(images)  # [batch_size, 64]
            elif isinstance(model, onnxruntime.InferenceSession):
                input_names = [i.name for i in model.get_inputs()]
                embeddings, _, _, _ = model.run(None, {input_names[0] : images.detach().cpu().numpy()})  # [batch_size, 64]
                embeddings = torch.from_numpy(embeddings).to(device)
            else:
                raise Exception()

            # 遍历 batch 中的每个样本
            for embedding, label in zip(embeddings, name_labels):
                if label.item() not in class_prototypes:
                    # 如果该类别还没有原型，初始化为一个全零向量
                    class_prototypes[label.item()] = torch.zeros_like(embedding)
                    class_counts[label.item()] = 0

                # 累加特征向量
                class_prototypes[label.item()] += embedding
                class_counts[label.item()] += 1

    # 计算每个类别的原型（均值）
    for label in class_prototypes:
        class_prototypes[label] /= class_counts[label]

    return class_prototypes