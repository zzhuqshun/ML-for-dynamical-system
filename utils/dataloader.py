import os
import numpy as np
import meshio
from tensorflow import keras
from scipy.interpolate import griddata
from sklearn.model_selection import train_test_split

def load_vtk_data(data_path, x, y, path_sep='\\'):
    """
    加载指定文件夹下的所有 .vtk 文件数据，输出归一化的 Qdot 数据和条件参数
    :param data_path: 数据目录（如 "./Data"）
    :param x: X方向网格数组
    :param y: Y方向网格数组
    :param path_sep: 路径分隔符，Windows用'\\', Linux/macOS用'/'
    :return: Qdot数组，条件参数数组
    """
    subfolder = []
    condition = []
    Q_list = []
    folders = os.listdir(data_path)

    # 遍历所有子文件夹
    for i in folders:
        if os.path.isdir(os.path.join(data_path, i)):
            subfolder.append(os.path.join(data_path, i))
    # 遍历所有子文件夹下的 .vtk 文件
    for folder in subfolder:
        files = os.listdir(folder)
        for file in files:
            ext = os.path.splitext(file)
            # 跳过 _ 结尾的文件
            if (ext[-1].lower() == '.vtk') and (ext[0][-2] != '_'):
                # 解析物理条件参数
                string = ext[0].replace('ER', '').replace('Tin', '').replace('Uin', '').replace('Twall', '').split('_')[0:4]
                var = []
                for j in string:
                    if j == 'Adiabatic':
                        var.append(0.)
                    else:
                        var.append(float(j))
                condition.append(var)
                # 读取vtk数据
                mesh = meshio.read(os.path.join(folder, file))
                points = mesh.points
                Qdot = mesh.point_data['Qdot']
                # 只取 y==0 并且 x>=0.006 的点
                boolArr = (points[:, 1] == 0) & (points[:, 0] >= 0.006)
                Qdot = Qdot[boolArr]
                points = points[boolArr]
                old_points = points[:, [0, 2]]
                grid_x, grid_y = np.meshgrid(x, y)
                # 插值到规则网格
                grid_new = griddata(old_points, Qdot, (grid_x, grid_y), method='nearest')
                Q_list.append(grid_new)
    Qdot_array = np.array(Q_list)
    condition_array = np.array(condition)
    return Qdot_array, condition_array

def remove_outliers(Qdot, condition, indices):
    """
    移除异常点（索引给定）
    """
    Qdot = np.delete(Qdot, indices, axis=0)
    condition = np.delete(condition, indices, axis=0)
    return Qdot, condition

def normalize_data(Qdot, condition):
    """
    数据归一化处理
    Qdot: 所有图片/样本的三维数据
    condition: 条件参数（标签）
    """
    # Qdot归一化
    Qdot_norm = Qdot / np.max(Qdot)
    Qdot_norm = np.reshape(Qdot_norm, (-1, 75, 300, 1))
    # 条件归一化
    normaliser = []
    df = np.zeros(condition.shape)
    for i in range(condition.shape[1]):
        df[:, i] = condition[:, i] / np.max(condition[:, i])
        normaliser.append(np.max(condition[:, i]))
    return Qdot_norm, df, normaliser

def split_data(Qdot, labels, test_size=0.15):
    """
    划分训练/测试集
    """
    train_data, test_data, label_train, label_test = train_test_split(
        Qdot, labels, test_size=test_size, shuffle=True
    )
    return train_data, test_data, label_train, label_test

def augment_data(train_data, label_train, times=5):
    """
    数据增强（每张图片生成 times 倍增强样本）
    """
    datagen = keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.1,
        fill_mode='nearest'
    )
    train_data_augmented = []
    label_train_augmented = []
    for i in range(train_data.shape[0]):
        img = train_data[i].reshape((1, 75, 300, 1))
        it = datagen.flow(img, batch_size=1)
        for _ in range(times):
            batch = it.next()
            train_data_augmented.append(batch[0])
            label_train_augmented.append(label_train[i])
    train_data_augmented = np.array(train_data_augmented)
    label_train_augmented = np.array(label_train_augmented)
    return train_data_augmented, label_train_augmented