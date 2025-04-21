import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

fashion = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion.load_data()
train_images = train_images.reshape(train_images.shape[0], -1)
test_images = test_images.reshape(test_images.shape[0], -1)
print(np.shape(train_images))
print(np.shape(test_images))
train_images_df = pd.DataFrame(train_images)
train_images_df.insert(0,'labels',train_labels)
print(train_images_df)
test_images_df = pd.DataFrame(test_images)
test_images_df.insert(0,'labels',test_labels)
print(test_images_df)

train_images_df.to_csv('fashion.csv', index=False)

# As expected, the CSV has the first column (A) with the labels (0-9) and the subsequent columns (B to ADD) contain the
# pixel values (0 to 255) for each of the flattened images. The data is mostly 0s, as expected, representing the large
# whitespace around the edges of the images. There are 785 columns in total, which makes sense because after
# subtracting 1 for the label column, that leaves 784, which is 28*28 and thus represents all the pixels in each image.

print("Shape of training dataframe is", np.shape(train_images_df))
print("Shape of testing dataframe is", np.shape(test_images_df))

# Target variable is in the form of numbers (0-9)

apparel_dict = {0:"T-shirt/top", 1:"Trouser", 2:"Pullover", 3:"Dress", 4:"Coat", 5:"Sandal",
                6:"Shirt", 7:"Sneaker", 8:"Bag", 9:"Ankle boot"}

train_images_original = train_images_df.copy()
train_images_df["labels"] = train_images_df["labels"].map(apparel_dict)
plt.figure()
sb.countplot(x=train_images_df["labels"])
plt.title('Count Plot for Training Data')
plt.show()

sample = train_images_df.sample(25) # i used sample to avoid repeats
sample_X = sample.iloc[:,1:]
sample_Y = sample.iloc[:,0]
plt.figure(figsize=(17,17))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    pixel = sample_X.iloc[i]
    pixel = np.array(pixel)
    pixel = pixel.reshape(28,28)
    plt.imshow(pixel, cmap=plt.cm.binary)
    plt.xlabel(str(sample_Y.iloc[i]))
plt.show()

X_train = train_images_original.iloc[:,1:]
Y_train = train_images_original.iloc[:,0]
X_test = test_images_df.iloc[:,1:]
Y_test = test_images_df.iloc[:,0]
X_train = X_train/255
X_test = X_test/255

model = keras.models.Sequential()
model.add(keras.Input(shape=(784,)))
model.add(keras.layers.Dense(200, activation = "leaky_relu"))
model.add(keras.layers.Dense(100, activation = "relu"))

model.add(keras.layers.Dense(10, activation = "softmax"))

model.summary()

model.compile(loss = "sparse_categorical_crossentropy",
              optimizer = "sgd", metrics = ["accuracy"])

h = model.fit(X_train, Y_train, epochs = 100, verbose=1)

pd.DataFrame(h.history).plot()
plt.show()

loss, accuracy = model.evaluate(X_test,Y_test)
print("Model accuracy on test data is", accuracy)

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_pred = [apparel_dict[i] for i in y_pred]
Y_test = Y_test.map(apparel_dict)
test_sample = np.array(X_test.iloc[0]).reshape(28, 28)
plt.imshow(test_sample, cmap="gray")
plt.xlabel("The predicted apparel is " + str(y_pred[0]))
plt.title("The actual apparel is " + str(Y_test.iloc[0]))
plt.show()

failed_df = X_test[y_pred != Y_test]
failed_index = failed_df.sample(n=1).index
failed_sample = np.array(X_test.iloc[failed_index]).reshape(28, 28)
failed_index = failed_index.item()
plt.imshow(failed_sample, cmap="gray")
plt.title("The failed predicted apparel is " + str(y_pred[failed_index]) +
          " whereas the actual apparel is " + str(Y_test[failed_index]))
plt.show()
