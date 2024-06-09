import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, precision_score, recall_score


# 模拟生成样本数据
def generate_data(num_samples, num_features, num_classes):
    X = np.random.randn(num_samples, num_features)
    y = np.random.randint(num_classes, size=num_samples)
    return X, y


# 自适应合成采样(ADASYN)算法
def adasyn(X, y, beta=1):
    # 计算每个类别的样本数量
    class_counts = np.bincount(y)

    # 计算总的少数类样本数量
    min_class_count = np.min(class_counts)

    # 计算每个少数类样本需要合成的新样本数量
    num_samples_to_generate = beta * (class_counts[0] - min_class_count)

    # 对每个少数类样本进行合成
    X_minority = X[y == 1]
    X_resampled = X_minority.copy()

    # 使用KNN算法找到每个少数类样本的k个最近邻居
    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(X)

    for i in range(len(X_minority)):
        # 找到当前少数类样本的k个最近邻居
        neighbors_indices = knn.kneighbors([X_minority[i]], return_distance=False)[0]

        # 从k个最近邻居中随机选择一个样本
        random_neighbor_index = np.random.choice(neighbors_indices)
        random_neighbor = X[random_neighbor_index]

        # 在当前少数类样本和随机选择的邻居之间随机插值生成新样本
        alpha = np.random.rand()
        new_sample = X_minority[i] + alpha * (random_neighbor - X_minority[i])

        # 将新生成的样本添加到重采样数据集中
        X_resampled = np.vstack((X_resampled, new_sample))

    # 将重采样的少数类样本与原始的多数类样本合并
    X_resampled = np.vstack((X_resampled, X[y == 0]))
    y_resampled = np.hstack((np.ones(len(X_resampled) - len(X[y == 0])), np.zeros(len(X[y == 0]))))

    return X_resampled, y_resampled


# 分裂卷积模块(SPC)
def split_conv(X, num_filters):
    # 将输入特征矩阵分裂成多个子特征矩阵
    num_features = X.shape[1]
    split_size = num_features // num_filters
    X_split = np.array(np.split(X, num_filters, axis=1))

    # 对每个子特征矩阵应用卷积操作
    X_conv = []
    for i in range(num_filters):
        # 使用1D卷积对子特征矩阵进行卷积操作
        conv_output = np.convolve(X_split[i].flatten(), np.ones(3), mode='same')
        X_conv.append(conv_output.reshape(-1, split_size))

    X_conv = np.concatenate(X_conv, axis=1)

    return X_conv


# AS-CNN模型
def as_cnn(X, y):
    # 数据预处理
    X_resampled, y_resampled = adasyn(X, y)

    # SPC-CNN特征提取
    X_features = split_conv(X_resampled, num_filters=4)

    # 分类器训练
    # 这里使用一个简单的阈值分类器作为示例
    threshold = 0.5
    y_pred = (np.mean(X_features, axis=1) > threshold).astype(int)

    # 模型评估
    accuracy = accuracy_score(y_resampled, y_pred)
    precision = precision_score(y_resampled, y_pred)
    recall = recall_score(y_resampled, y_pred)

    return accuracy, precision, recall


# 主函数
def main():
    # 生成样本数据
    X, y = generate_data(num_samples=1000, num_features=100, num_classes=2)

    # 应用AS-CNN模型进行入侵检测
    accuracy, precision, recall = as_cnn(X, y)

    # 输出性能指标
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    print("Precision: {:.2f}%".format(precision * 100))
    print("Recall: {:.2f}%".format(recall * 100))


if __name__ == '__main__':
    main()