from keras.datasets import mnist
from autokeras import ImageClassifier

# loadning mnist from keras
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# each image has to 3D: 2 coordinates, 1 value (gray scale)
X_train = X_train.reshape(X_train.shape + (1,))
X_test = X_test.reshape(X_test.shape + (1,))

clf = ImageClassifier(verbose=True, augment=True)
clf.fit(X_train, y_train, time_limit=(1*60*60))