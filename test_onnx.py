import ast
import torch
from torchvision import transforms
from test import predict
from myDataset import MyDataset
import pandas as pd
import onnxruntime

if __name__ == '__main__':
    # 检查是否有可用的 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    session = onnxruntime.InferenceSession('model.onnx')
    prefix_list = ast.literal_eval(session.get_modelmeta().custom_metadata_map['prefix_list'])

    import base64
    import numpy
    df = pd.read_csv('训练集原型特征.csv')
    class_prototypes : list[list[str]] = df.values.tolist()

    class_prototypes = {row[0]: torch.tensor(numpy.frombuffer(base64.b64decode(row[1].encode("utf-8")), dtype=numpy.float32), requires_grad=False).to(device) for row in class_prototypes}
    
    df = pd.read_csv('测试集列表.csv', header=None)
    test_set : list[list[str]] = df.values.tolist()
    datasetTest = MyDataset([x for row in test_set for x in row])

    df_pred = pd.DataFrame()
    embeddings = []
    with torch.no_grad():
        for t, name_label, prefix_label, star_num in datasetTest:
            name = datasetTest.label_prefix_dict[prefix_label] + datasetTest.label_name_dict[name_label] + '★' * star_num
            print(f'应该是：{name}')
            # 提取特征向量
            embedding : torch.Tensor
            embedding, distance, pred_name, pred_prefix_label, pred_star_num = predict(session, class_prototypes, t, device)
            embeddings.append(embedding.squeeze().detach().cpu().numpy())

            name_predict = prefix_list[pred_prefix_label] + pred_name + '★' * pred_star_num
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

