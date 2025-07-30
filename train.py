import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from myDataset import MyDataset, get_triplets, split_into_train_test
from myLoss import DualTripletMarginLoss
from myModules import PrototypicalNetwork, compute_class_prototypes
from torch.utils.data import DataLoader

if __name__ == '__main__':
    # 数据预处理
    transform = transforms.Compose([
                    transforms.Resize((153, 125)),
                    transforms.ToTensor()
                ])

    # 创建 Dataset
    root_dir = r'爆炒肉片new'
    train_set, test_set = split_into_train_test(root_dir, 10, 5, 1)
    dataset = MyDataset([t[0] for t in train_set], transform=transform)

    # 创建 DataLoader
    batch_size = len(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    num_epochs = 100

    # 初始化模型、损失函数和优化器
    model = PrototypicalNetwork()
    # criterion = nn.TripletMarginLoss(margin=20, p=2)  # TripletMarginLoss
    criterion = DualTripletMarginLoss(margin_pos=3, margin_neg=10)  # 修改的三元组损失函数
    prefix_loss_fn = nn.CrossEntropyLoss();
    star_loss_fn = nn.CrossEntropyLoss();
    optimizer = optim.Adam([
        {'params' : model.conv1.parameters()},
        {'params' : model.star_fc.parameters(), 'lr' : 1e-1},
        {'params' : model.conv2.parameters()},
        {'params' : model.conv3.parameters()},
        {'params' : model.fc2.parameters()},
        {'params' : model.prefix_fc.parameters()},
        # {'params' : model.prefix_lambda, 'lr' : 1e-1},
        # {'params' : model.star_lambda, 'lr' : 1e-1},
        ], lr = 1e-2)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=batch_size, eta_min=1e-4);
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=15, factor=0.5);

    # 检查是否有可用的 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, name_labels, prefix_labels, star_nums) in enumerate(dataloader):
            images, name_labels, prefix_labels, star_nums = images.to(device), name_labels.to(device), prefix_labels.to(device), star_nums.to(device)
        
            # 前向传播
            embeddings, prefix_logists, star_logists = model(images)

            # 手动构造三元组
            anchor_embeddings, positive_embeddings, negative_embeddings = get_triplets(embeddings, name_labels)

            # 如果无法构造三元组（比如某些类别样本不足），跳过该 batch
            if anchor_embeddings is None:
                print(f"Skipping batch {batch_idx} due to insufficient triplets.")
                continue

            # 计算损失
            name_loss, margin = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
            prefix_loss = prefix_loss_fn(prefix_logists, prefix_labels)
            star_loss = star_loss_fn(star_logists, star_nums)
            print(f'NameLoss: {name_loss.item() / len(dataloader):.4f}, Margin: {margin:.4f}, PrefixLoss: {prefix_loss.item() / len(dataloader):.4f}, StarLoss: {star_loss.item() / len(dataloader):.4f}')
            
            loss = 0.4 * name_loss + 0.3 * prefix_loss + 0.3 * star_loss
        
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            # scheduler.step()
            # if epoch > 5:   # 忽略前几轮波动
            #     scheduler.step(loss)

        # print(f'Prefix lambda: {model.prefix_lambda.item()}, Star lambda: {model.star_lambda.item()}')
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}, Lr: {optimizer.param_groups[0]['lr']:.6f}")
        if running_loss < 1e-2:
            print('哇塞！是百分之一loss！应该可以了')
            break


    with torch.no_grad():
        last = {
            "model_state_dict": model.state_dict(),
            "label_prefix_dict": dataset.label_prefix_dict
        }
        torch.save(last, "model.pth")

        dummy_input = torch.rand(1, 3, 153, 125).to(device)
        torch.onnx.export(model, dummy_input, "model.onnx", input_names=['input'], dynamic_axes={'input' : {0 : 'batch_size'}})
        import onnx
        onnx_model = onnx.load("model.onnx")
        meta = onnx_model.metadata_props.add()
        meta.key = "label_prefix_dict"
        meta.value = str(dataset.label_prefix_dict).replace("'", '"')
        onnx.save(onnx_model, "model.onnx")

        import numpy
        import pandas as pd
        import base64
        df_train = pd.DataFrame()
        df_train['标注名称'] = [tp[2] + tp[1] + '★' * tp[3] for tp in dataset.data]

        embeddings = []        
        embeddings_base64 = []
        dataloader2 = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for batch_idx, (images, _, _, _) in enumerate(dataloader2):
            images = images.to(device)
            batch_embeddings, _, _ = model(images)
            batch_embeddings = [e.detach().cpu().numpy() for e in batch_embeddings]
            batch_embeddings : list[numpy.ndarray]
            embeddings.extend(batch_embeddings)
            embeddings_base64.extend([base64.b64encode(eb.tobytes()).decode("utf-8") for eb in batch_embeddings])

        # 保存为本地的 npy 文件，用于对比文件大小
        numpy.save('训练集样本特征.npy', embeddings)

        # 保存所有样本向量
        df_train['特征向量'] = embeddings_base64
        df_train.to_csv('训练集样本特征.csv', index=False)

        # 保存原型向量
        class_prototypes : dict[int, torch.Tensor] = compute_class_prototypes(model, dataloader, device)
        df_prot = pd.DataFrame()
        df_prot['标注名称'] = [dataset.label_name_dict[label] for label in class_prototypes.keys()]
        df_prot['特征向量'] = [base64.b64encode(t.detach().cpu().numpy().tobytes()).decode("utf-8") for t in class_prototypes.values()]
        df_prot.to_csv('训练集原型特征.csv', index=False)

        df_test = pd.DataFrame([t[0] for t in test_set])
        df_test.to_csv('测试集列表.csv', index=False, header=False)

