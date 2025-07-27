import matplotlib
import matplotlib.pyplot as plt
from umap import UMAP
from myDataset import getStar
from visualizeModel import drawFeatureMatrix, getStarColor, loadFeatureMatrix
import numpy as np

if __name__ == '__main__':
    matplotlib.rc("font",family='SimHei') # 中文字体
    plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号
    
    
    mapper :UMAP = UMAP(n_neighbors=10, n_components=2, random_state=233)
    figure = plt.figure(figsize=(16, 9))
    
    encoding_array, df = loadFeatureMatrix('模型特征.npy', '模型特征.csv')    
    encoding_array_pred, df_pred = loadFeatureMatrix('测试集特征.npy', '测试集预测结果.csv')

    # 合并进行umap训练(fit)效果会好一些，只用训练集特征进行fit再用测试集特征transform效果较差
    X_umap_2d = mapper.fit_transform(np.vstack((encoding_array, encoding_array_pred)))
    print(X_umap_2d.shape)
    
    drawFeatureMatrix(X_umap_2d, df);
    figure.legend().remove();

    scatters = []
    for idx, row in df_pred.iterrows(): # 遍历每个类别
        x = X_umap_2d[idx + len(encoding_array), 0]
        y = X_umap_2d[idx + len(encoding_array), 1]
        _, star = getStar(row['标注名称'])
        color = getStarColor(star)
        scatter = plt.scatter(x, y, label=row['标注名称'], color=color, s=100, marker=',')
        scatters.append(scatter)
        if row['标注名称'] == row['预测名称']:
            plt.text(x, y, row['标注名称'], {'color' : 'green'}, rotation=15)
        else:
            plt.text(x, y, f"{row['标注名称']}/预测成了：{row['预测名称']}", {'color' : 'red'}, rotation=15)

    plt.legend(fontsize=10, markerscale=1, bbox_to_anchor=(1, 1), handles=scatters)
    plt.xticks([])
    plt.yticks([])
    plt.title('测试集预测结果')
    plt.show()