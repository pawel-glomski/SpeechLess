# %%
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
import pickle
import GenerateDataset
import soundfile as sf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def plot_history(history):
    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.savefig("LearningHistory.jpg")


def loadDataset(validationRatio):
    dataset = pickle.load(dsFile)
    dataset["MFCC"] = dataset["MFCC"][..., np.newaxis]
    temp = dataset["range"][0]
    return (dataset, (temp[1] - temp[0], *dataset["MFCC"].shape[1:]))


def makeModel(inputShape):
    model = keras.Sequential()

    # model.add(keras.layers.LSTM(128, input_shape=inputShape, return_sequences=True,
    #                             kernel_regularizer=keras.regularizers.l2(0.001)))
    # model.add(keras.layers.LSTM(128, return_sequences=True,
    #                             kernel_regularizer=keras.regularizers.l2(0.001)))
    # model.add(keras.layers.Reshape((inputShape[0], 128, 1)))

    model.add(keras.layers.Conv2D(64, (5, 3), activation="relu", input_shape=inputShape,
                                  kernel_regularizer=keras.regularizers.l2(0.001)
                                  ))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=inputShape,
                                  kernel_regularizer=keras.regularizers.l2(0.001)
                                  ))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu',
                                 kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


def splitDataset(dataset, validationPart):
    return train_test_split(dataset["range"], dataset["label"], test_size=validationPart)


def prepareBatch(batch, dataset):
    return np.array([dataset["MFCC"][segRange[0]:segRange[1]] for segRange in batch])


def fit(model: keras.Model, dataset, batchSize, epochs, validationPart=0.2):
    x, vX, y, vY = splitDataset(dataset, validationPart)

    hist = type('', (), {})  # mock object
    setattr(hist, "history", {"accuracy": [],
                              "loss": [],
                              "val_accuracy": [],
                              "val_loss": []})

    epochLen = int(x.shape[0] / batchSize)
    for e in range(epochs):
        # train
        for i, trainIdcs in enumerate(np.random.permutation(epochLen * batchSize).reshape(epochLen, batchSize)):
            trainX = prepareBatch(np.take(x, trainIdcs, axis=0), dataset)
            trainY = np.take(y, trainIdcs, axis=0)
            trLoss, trAcc = model.train_on_batch(trainX, trainY, reset_metrics=False)
            print("Epoch #{}: {:.2f}%\r".format(e+1, i / epochLen * 100), end="", flush=True)
        model.reset_metrics()
        # validate
        for i in range(batchSize):
            valIdcs = np.random.choice(vX.shape[0], batchSize)
            valX = prepareBatch(np.take(vX, valIdcs, axis=0), dataset)
            valY = np.take(vY, valIdcs, axis=0)
            vaLoss, vaAcc = model.test_on_batch(valX, valY, reset_metrics=False)
        model.reset_metrics()

        print("Epoch #{}: Done! trAcc = {:.4f}, trLoss = {:.4f}, vaAcc = {:.4f}, vaLoss = {:.4f}".
              format(e+1, trAcc, trLoss, vaAcc, vaLoss))

        hist.history["loss"].append(trLoss)
        hist.history["accuracy"].append(trAcc)
        hist.history["val_loss"].append(vaLoss)
        hist.history["val_accuracy"].append(vaAcc)
    return hist


with open(GenerateDataset.DATASET_PATH, "rb") as dsFile:
    dataset, inputShape = loadDataset(0.2)
    model = makeModel(inputShape)
    plot_history(fit(model, dataset, 32, 60))
    model.save("lastModel")

    # model = keras.models.load_model("lastModel")
    # testSig = GenerateDataset.loadSignal("/mnt/Shared/Code/Python/LectureCutter/data/mn2.labels")
    # mfcc = GenerateDataset.calcMFCCs(testSig)
    # outBad = []
    # outGood = []
    # lastMFCCIdx = len(mfcc) - GenerateDataset.SEG_LEN
    # for i in range(0, lastMFCCIdx, GenerateDataset.SEQ_LEN):
    #     segMFCC = (mfcc[i:i+GenerateDataset.SEG_LEN])[np.newaxis, ..., np.newaxis]
    #     idx = (i+GenerateDataset.SEQ_PADDING)*GenerateDataset.HOP_LENGTH
    #     seqLen = GenerateDataset.SEQ_LEN * GenerateDataset.HOP_LENGTH
    #     if model.predict(segMFCC) < 0.3:
    #         outBad = np.concatenate((outBad, testSig[idx:idx+seqLen]))
    #     else:
    #         outGood = np.concatenate((outGood, testSig[idx:idx+seqLen]))
    #     print("{:.2f} %".format(i/lastMFCCIdx * 100))
    # sf.write("bad.wav", outBad, 22050, 'PCM_16', format="WAV")
    # sf.write("good.wav", outGood, 22050, 'PCM_16', format="WAV")
