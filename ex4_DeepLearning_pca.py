import matplotlib
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from keras.utils import to_categorical
import tensorflow as tf
from sklearn.metrics import classification_report

iris_idex = datasets.load_iris()    # データロード
x = iris_idex.data

#主成分分析の実行
num = 2 #圧縮後の次元数
pca = PCA(n_components=num)
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

#機械学習に使用するデータを任意に分割
X_train, X_test, y_train, y_test = train_test_split(iris_pca[["com1","com2"]],iris_pca["target"], test_size=0.2, random_state=1)

#目的変数のone-hotエンコーディング
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# モデルの構築
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(num), #PCAで圧縮後の次元数=num
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
 
#学習の実行
result = model.fit(X_train, y_train, epochs=100, batch_size=25,validation_data=(X_test, y_test))

# Accuracyのプロット
plt.figure()
plt.title('Accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.plot(result.history['accuracy'], label='train')
plt.plot(result.history['val_accuracy'], label='test')
plt.legend()

# Lossのプロット
plt.figure()
plt.title('categorical_crossentropy Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(result.history['loss'], label='train')
plt.plot(result.history['val_loss'], label='test')
plt.legend()
plt.show()

#学習モデルを使った予測結果
iris_pred = model.predict(X_test)
iris_pred = np.argmax(iris_pred, axis=1)
print(iris_pred)

#学習モデルの精度
score = model.evaluate(X_test,y_test,batch_size=1)
print('y_test accuracy:', score[1])

#学習モデルの評価
print(classification_report(y_test, to_categorical(iris_pred)))