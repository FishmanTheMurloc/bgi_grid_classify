import matplotlib
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from umap import UMAP
from myDataset import getStar

def getStarColor(star):
    if star == 0 or star == '' or star == 1 or star == '★':
        return '#727376'
    elif star == 2 or star == '★★':
        return '#51876f'
    elif star == 3 or star == '★★★':
        return '#537c9c'
    elif star == 4 or star == '★★★★':
        return '#8167a6'
    elif star == 5 or star == '★★★★★':
        return '#b57227'
    else:
        return '#000000'

def loadFeatureMatrix(npy :str, csv :str) -> tuple[np.ndarray, pd.DataFrame]:
    encoding_array = np.load(npy, allow_pickle=True)
    print(encoding_array.shape)

    df = pd.read_csv(csv)
    print(df.head())
    return encoding_array, df

def drawFeatureMatrix(X_umap_2d, df:pd.DataFrame):
    for idx, row in df.iterrows(): # 遍历每个类别
        x = X_umap_2d[idx, 0]
        y = X_umap_2d[idx, 1]
        _, star = getStar(row['标注名称'])
        color = getStarColor(star)
        plt.scatter(x, y, label=row['标注名称'], color=color, s=100)
        plt.text(x, y, row['标注名称'])

if __name__ == '__main__':
    matplotlib.rc("font",family='SimHei') # 中文字体
    plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号

    # plt.plot([1,2,3], [100,500,300])
    # plt.title('matplotlib中文字体测试', fontsize=25)
    # plt.xlabel('X轴', fontsize=15)
    # plt.ylabel('Y轴', fontsize=15)
    # plt.show()

    encoding_array, df = loadFeatureMatrix('模型特征.npy', '模型特征.csv')
    
    mapper :UMAP = UMAP(n_neighbors=10, n_components=2, random_state=233)
    figure = plt.figure(figsize=(16, 9))
    
    X_umap_2d = mapper.fit_transform(encoding_array)
    print(X_umap_2d.shape)

    drawFeatureMatrix(X_umap_2d, df);

    plt.legend(fontsize=10, markerscale=1, bbox_to_anchor=(1, 1))
    plt.xticks([])
    plt.yticks([])
    plt.title('模型特征')
    plt.show()