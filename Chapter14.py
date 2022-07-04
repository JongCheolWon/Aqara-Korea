import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dropout(0.2), tf.keras.layers.Dense(10, activation='softmax')])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# 에포트 횟수가 반복될수록 비용함수는 줄어들고 정확도는 향상되고 있다.
hist = model.fit(x_train, y_train, epochs=5)


# 최종 인식 정확도는 98.1%로 도출되었다.
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('test_acc:', test_acc)

model.summary()


import matplotlib as mpl
import matplotlib.pylab as plt

plt.figure(figsize=(8,4))
plt.subplot(1, 2, 1)
plt.plot(hist.history['loss'])
plt.title("loss")
plt.subplot(1, 2, 2)

plt.title("accuracy")
plt.plot(hist.history['accuracy'], 'b-', label="training")
# plt.plot(model.history['val_accuracy'], 'r:', label="validation")
plt.legend()
plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pylab as plt
from skimage.data import coffee
from skimage.filters import sobel_h, sobel_v
from skimage.color import rgb2gray

# 커피 이미지를 INPUT
image = coffee()
grey_image = rgb2gray(image)

# CONV
edge_h = sobel_h(grey_image)
edge_v = sobel_v(grey_image)

# ReLU
edge_h = np.where(edge_h < 0, 0, edge_h)
edge_v = np.where(edge_v < 0, 0, edge_v)
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(image)
plt.xticks([])
plt.yticks([])
plt.subplot(1,3,3)
plt.imshow(edge_v, cmap=plt.cm.gray)
plt.xticks([])
plt.yticks([])
plt.show()

# 각각 필터링 된 이미지를 비교해 보자.


from tensorflow.keras import datasets, layers, models
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# 픽셀값을 0~1 사이로 정규화한다.
train_images, test_images = train_images / 255.0, test_images / 255.0


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))


model.summary()


model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


model.summary()


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# 에포트 횟수가 반복될수록 손실함수는 줄어들고 정확도는 향상되고 있다.
model.fit(train_images, train_labels, epochs=5)


# 최종 인식 정확도는 99.1%로 나타났다.
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('test_acc:', test_acc)


import matplotlib as mpl
import matplotlib.pylab as plt

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(hist.history['loss'])
plt.title("loss")
plt.subplot(1, 2, 2)

plt.title("accuracy")
plt.plot(hist.history['accuracy'], 'b-', label="training")
#plt.plot(model.history['val_accuracy'], 'r:', label="validation")
plt.legend()
plt.tight_layout()
plt.show()


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
train_path = tf.keras.utils.get_file("iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file("iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)


train.head()


train_y = train.pop('Species')
test_y = test.pop('Species')
train.head()


def input_evaluation_set():
    		features = {'SepalLength': np.array([6.4, 5.0]), 'SepalWidth': np.array([2.8, 2.3]), 'PetalLength': np.array([5.6, 3.3]), 'PetalWidth': np.array([2.2, 1.0])}
		labels = np.array([2, 1])
		return features, labels

def input_fn(features, labels, training=True, batch_size=256):
		"""An input function for training or evaluating"""
		# Convert the inputs to a Dataset.
		dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

		# Shuffle and repeat if you are in training mode.
		if training:
				dataset = dataset.shuffle(1000).repeat()

		return dataset.batch(batch_size)

# Feature columns describe how to use the input.
my_feature_columns = []
for key in train.keys():

my_feature_columns.append(tf.feature_column.numeric_column(key=key))


# 30, 10 유닛으로 구성된 2개 계층 DNN 생성
classifier = tf.estimator.DNNClassifier(feature_columns=my_feature_columns, hidden_units=[30, 10], n_classes=3) # 3개 클래스 다중 분류


classifier.train(input_fn=lambda: input_fn(train, train_y, training=True), steps=5000)


eval_result = classifier.evaluate(input_fn=lambda: input_fn(test, test_y, training=False))
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


# X값 입력으로 모델 예측
# Prediction is "Setosa" (84.3%), expected "Setosa"
# Prediction is "Versicolor" (46.5%), expected "Versicolor"
# Prediction is "Virginica" (59.5%), expected "Virginica"

expected = ['Setosa', 'Versicolor', 'Virginica']
predict_x = {'SepalLength': [5.1, 5.9, 6.9], 'SepalWidth': [3.3, 3.0, 3.1], 'PetalLength': [1.7, 4.2, 5.4], 'PetalWidth': [0.5, 1.5, 2.1]}

def input_fn(features, batch_size=256):
		"""An input function for prediction."""
		# Convert the inputs to a Dataset without labels.
		return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

predictions = classifier.predict(input_fn=lambda: input_fn(predict_x))

for pred_dict, expec in zip(predictions, expected):
		class_id = pred_dict['class_ids'][0]
		probability = pred_dict['probabilities'][class_id]

		print('Prediction is "{}" ({:.1f}%), expected "{}"'.format(SPECIES[class_id], 100 * probability, expec))
