import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report


# 假设你有一个多标签分类的数据集X和对应的标签y
# X是特征矩阵，y是标签矩阵，每行对应一个样本，每列对应一个标签

# 分割数据集为训练集和测试集

X = pd.read_csv('./data/OA/train_data.csv')
y = pd.read_csv('./data/OA/train_label.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建随机森林分类器
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=0)

# 构建多标签分类器
multi_label_classifier = MultiOutputClassifier(rf_classifier, n_jobs=-1)

# 训练多标签分类器
multi_label_classifier.fit(X_train, y_train)

# 预测标签
y_pred = multi_label_classifier.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)

# 计算精确度
precision = precision_score(y_test, y_pred, average='weighted')

# 计算召回率
recall = recall_score(y_test, y_pred, average='weighted')

# 计算F1分数
f1 = f1_score(y_test, y_pred, average='weighted')

# 使用classification_report生成详细的分类报告，包括精确度、召回率、F1分数和支持度
report = classification_report(y_test, y_pred)

print("精确度:", precision)
print("召回率:", recall)
print("F1分数:", f1)
print("分类报告:")
print(report)