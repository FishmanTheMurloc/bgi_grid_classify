import os
import torch
from torch.utils.data import dataset
from torchvision import transforms
from torch import Tensor
from PIL import Image
import pathlib

class MyDataset(dataset.Dataset):
    label_name_dict : dict
    '''
    根据名称标签查找名称的字典
    '''
    label_prefix_dict : dict
    '''
    根据前缀标签查找前缀的字典
    0: 无前缀, 1: 美味的, 2: 奇怪的
    '''

    def __init__(self, file_paths:list[str], transform = None):
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                    transforms.Lambda(self.conditional_resize),
                    # transforms.Resize((153, 125)),
                    transforms.Lambda(lambda x: x.crop((0, 0, 125, 125))),  # 裁剪左上角的125x125
                    transforms.ToTensor()
                ])

        self.data = list[tuple]()
        self.name_label_dict = dict[str, int]()
        self.prefix_label_dict : dict[str, int] = {'': 0, '美味的': 1, '奇怪的': 2}
        for file_path in file_paths:
            path_obj = pathlib.Path(file_path)

            first_subdir = path_obj.parts[1] if len(path_obj.parts) > 2 else None   # 第一级子文件夹名
            game_source_name = getGameSourceName(first_subdir)
            is_food = game_source_name == 'Food'

            prefix, item_name = getPrefix(path_obj.stem)
        
            item_name, star = getStar(item_name)
            star_num = len(star)
            
            if item_name not in self.name_label_dict:   # 名称的标签就是从0开始自增啦
                self.name_label_dict[item_name] = len(self.name_label_dict)
            self.data.append((file_path, item_name, prefix, star_num, is_food))
        self.label_name_dict = {v: k for k, v in self.name_label_dict.items()}
        self.label_prefix_dict = {v: k for k, v in self.prefix_label_dict.items()}

    def __getitem__(self, index) -> tuple[Tensor, str, str, str, bool]:
        file_path, item_name, prefix, star_num, is_food = self.data[index]

        image = Image.open(file_path).convert('RGB')
        tensor = self.transform(image)

        return tensor, self.name_label_dict[item_name], self.prefix_label_dict[prefix], star_num, is_food

    def __len__(self):
        return len(self.data)

    def conditional_resize(self, img):
        width, height = img.size    
        if width == 125 and height == 125:
            return img
        else:
            return transforms.functional.resize(img, (153, 125))


def getPrefix(item_name : str) -> tuple[str, str]:
    """
    获取'美味的'、'奇怪的'前缀
    Returns:
        prefix: 前缀
        rest: 剩余字符串
    """
    if item_name.startswith('美味的'):
        return '美味的', item_name[3:]
    elif item_name.startswith('奇怪的'):
        return '奇怪的', item_name[3:]
    elif any(s in item_name for s in ['雾凇秋分', '雾松秋分', '一捧绿野', '鎏金殿堂', '白浪拂沙']):  # 特色料理 自带美味特效但没有美味前缀
        return '美味的', item_name
    else:
        return '', item_name

def getStar(item_name : str) -> tuple[str, str]:
    """
    获取星星后缀
    Returns:
        rest: 剩余字符串
        star: 星星字符串
    """     
    without_star = item_name.rstrip('★');
    star = item_name[len(without_star):]
    return without_star, star

def getGameSourceName(folder_name : str|None) -> str|None:
    """
    获取文件夹名中的游戏来源名
    """
    if not folder_name:
        return None
    elif folder_name.startswith('Weapons'):
        return 'Weapons'
    elif folder_name.startswith('Artifacts'):
        return 'Artifacts'
    elif folder_name.startswith('CharacterDevelopmentItems'):
        return 'CharacterDevelopmentItems'
    elif folder_name.startswith('Food'):
        return 'Food'
    elif folder_name.startswith('Materials'):
        return 'Materials'
    elif folder_name.startswith('Gadget'):
        return 'Gadget'
    elif folder_name.startswith('Quest'):
        return 'Quest'
    elif folder_name.startswith('PreciousItems'):
        return 'PreciousItems'
    elif folder_name.startswith('Furnishings'):
        return 'Furnishings'
    elif folder_name.startswith('ArtifactSalvage'):
        return 'ArtifactSalvage'
    elif folder_name.startswith('ArtifactSetFilter'):
        return 'ArtifactSetFilter'
    else:
        return None

def get_triplets(embeddings, labels):
    """
    构造三元组 (anchor, positive, negative)
    Args:
        images: Tensor, shape [batch_size, 3, 125, 125]
        labels: Tensor, shape [batch_size]
    Returns:
        anchor, positive, negative: 三个 Tensor，形状均为 [num_triplets, 3, 125, 125]
    """
    anchor_list, positive_list, negative_list = [], [], []
    
    # 计算 pairwise 距离矩阵（欧氏距离平方）
    pairwise_dist = torch.cdist(embeddings, embeddings, p=2) ** 2  # (batch_size, batch_size)

    # 遍历 batch 中的每个样本
    for i in range(len(labels)):
        anchor_label = labels[i]

        # 1. 选择一个 positive 样本（与 anchor 同类别）
        positive_indices = [j for j in range(len(labels)) if labels[j] == anchor_label and j != i]
        if not positive_indices:
            # 如果没有 positive 样本（可能只有一个类别的样本），强行设置anchor=positive
            hardest_positive_idx = i
        else:
            pos_distances = pairwise_dist[i, positive_indices]
            hardest_positive_idx = positive_indices[torch.argmax(pos_distances)]
        positive_list.append(embeddings[hardest_positive_idx])

        # 2. 选择一个 negative 样本（与 anchor 不同类别）
        negative_indices = [j for j in range(len(labels)) if labels[j] != anchor_label]
        if not negative_indices:
            # 如果没有 negative 样本（可能只有一个类别），跳过该 anchor
            continue
        neg_distances = pairwise_dist[i, negative_indices]
        hardest_negative_idx = negative_indices[torch.argmin(neg_distances)]    # 选最难的
        negative_list.append(embeddings[hardest_negative_idx])

        # 3. anchor 就是当前样本
        anchor_list.append(embeddings[i])

    # 如果没有找到任何三元组（可能是因为某些类别样本太少），返回空列表
    if not anchor_list:
        return None, None, None

    # 将列表转换为 Tensor
    anchors = torch.stack(anchor_list)
    positives = torch.stack(positive_list)
    negatives = torch.stack(negative_list)

    return anchors, positives, negatives


import random
def split_into_train_test(root_dir : str, single_sample_num : int, dual_sample_num : int, triple_sample_num : int) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """
    把数据集划分为训练和测试
    Args:
        single_sample_num: 使用单样本的数量，既做训练，又做测试
        dual_sample_num: 使用双样本的数量，取出一个做测试，另一个做训练
        triple_sample_num: 使用三样本的数量，取出一个做测试，其余做训练
    Returns:
        train: 训练集 list[(file_path, name)]
        test: 测试集 list[(file_path, name)]
    """
    file_paths = [str(file) for file in pathlib.Path(root_dir).rglob('*.png')]

    sample_groups = list[tuple[str, list[str]]]()   # list[(name, list[file_path])]
    
    for file_path in file_paths:
        fname = os.path.basename(file_path)

        item_full_name = os.path.splitext(fname)[0] # 去文件扩展名
        _, item_name = getPrefix(item_full_name)
        
        item_name, _ = getStar(item_name)

        group = next((g for g in sample_groups if item_name == g[0]), None)    # https://www.markheath.net/post/python-equivalents-of-linq-methods
        if group:
            group[1].append(file_path)
        else:
            group = (item_name, [file_path])
            sample_groups.append(group)
      
    train = list[tuple[str, str]]() # list[(file_path, name)]
    test = list[tuple[str, str]]()

    single_sample_groups = [g for g in sample_groups if len(g[1]) == 1]
    random.shuffle(single_sample_groups)
    assert len(single_sample_groups) > single_sample_num
    test += [(p, g[0]) for g in single_sample_groups[:single_sample_num] for p in g[1]] # equivalent to SeletMany()
    train += [(p, g[0]) for g in single_sample_groups for p in g[1]]

    dual_sample_groups = [g for g in sample_groups if len(g[1]) == 2]
    random.shuffle(dual_sample_groups)
    assert len(dual_sample_groups) > dual_sample_num
    for group in dual_sample_groups[:dual_sample_num]:
        seleted = group[1][random.randint(0, 1)]
        test.append((seleted, group[0]))
        group[1].remove(seleted)
        train += [(p, group[0]) for p in group[1]]
    train += [(p, g[0]) for g in dual_sample_groups[dual_sample_num:] for p in g[1]]

    triple_sample_groups = [g for g in sample_groups if len(g[1]) == 3]
    random.shuffle(triple_sample_groups)
    assert len(triple_sample_groups) > triple_sample_num
    for group in triple_sample_groups[:triple_sample_num]:
        seleted = group[1][random.randint(0, 2)]
        test.append((seleted, group[0]))
        group[1].remove(seleted)
        train += [(p, group[0]) for p in group[1]]
    train += [(p, g[0]) for g in triple_sample_groups[triple_sample_num:] for p in g[1]]

    for group in [g for g in sample_groups if len(g[1]) > 3]:
        print(f'{group[0]}样本数大于3：')
        for file_path in group[1]:
            print(file_path)

    assert len(single_sample_groups) + len(dual_sample_groups) * 2 + len(triple_sample_groups) * 3 == len(file_paths), "难道还有大于3个样本的？"
    assert len(train) + len(test) - single_sample_num == len(file_paths)

    return train, test
    
if __name__ == '__main__':
    # ----测试MyDataset读取-----
    root_dir = r'爆炒肉片new'
    file_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir)
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
    dataset = MyDataset(file_paths, None);
    for t, name_label, prefix_label, star_num, is_food in dataset:
        print(f'{t.size()}, {name_label}, {prefix_label}, {star_num}, {is_food}')
    
    # ----测试分割数据集-----
    _, test = split_into_train_test(r'爆炒肉片new', 10, 5, 1)
    for sample in test:
        print(f'{sample[1]} => {sample[0]}')
    dataset = MyDataset([t[0] for t in test], None);
    for t, name_label, prefix_label, star_num, is_food in dataset:
        print(f'{t.size()}, {name_label}, {prefix_label}, {star_num}, {is_food}')
