import matplotlib
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

iris_idex = datasets.load_iris()    # データロード
x = iris_idex.data

#主成分分析の実行
pca = PCA(n_components=2)
feature = pca.fit_transform(x)

# 主成分得点
iris_pca = pd.DataFrame(feature, columns=["com1", "com2"])
iris_pca["target"] = iris_idex.target

# 色のリストを作成
colors = ["red", "green", "blue"]

# 第一主成分と第二主成分でプロットする
plt.figure(figsize=(6, 6))
plt.scatter(iris_pca["com1"], iris_pca["com2"], alpha=0.8, c=iris_pca["target"], cmap=matplotlib.colors.ListedColormap(colors))
plt.colorbar(label='Target Class')
plt.grid()
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

#テストデータを10-90%まで変化させたときのそれぞれの平均判別エラー値を求める
#学習回数は50回とする
num = 50 #判別回数
all_average_error = []
for size in np.arange(0.1, 1, 0.1):
  average_error = []
  for i in range(num):
    X_train, X_test, y_train, y_test = train_test_split(iris_pca[["com1","com2"]],iris_pca["target"], test_size=size, shuffle=True)
    model = GaussianNB()  #訓練データの学習
    model.fit(X_train, y_train) #判別関数の決定
    predicted = model.predict(X_test) #判別関数を適用して予測
    average_error.append(1 - metrics.accuracy_score(y_test, predicted))  #予測結果のエラー値
  print("テストデータ割合が" + str(math.floor(size*100)) + "%のとき，" + 
        str(num) + "回学習した時の平均判別エラー値は" + str(sum(average_error)/num) + "です．")
  all_average_error.append(sum(average_error)/num)

X_test_size = np.arange(0.1, 1, 0.1)
df = pd.DataFrame(all_average_error, columns=["average_error"], index=X_test_size)

plt.plot(X_test_size, all_average_error, color="b")
plt.xlabel("test_size")
plt.ylabel("average_error")
plt.xticks(np.arange(0.0, 1.0, 0.1))
plt.hlines([min(all_average_error)],0, 1, linestyles="dashed")
plt.show()