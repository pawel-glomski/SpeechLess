from sklearn.model_selection import train_test_split
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
import soundfile as sf
import numpy as np
import keras
import pickle
import GenerateDataset
import subprocess
import shutil

TO_OPTIMIZE_DIR = (Path.cwd()/("../toOptimize" if Path.cwd().name == "src" else "toOptimize")).resolve()

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
    dataset["data"] = dataset["data"][..., np.newaxis]
    temp = dataset["range"][0]
    return (dataset, (temp[1] - temp[0], *dataset["data"].shape[1:]))


def makeModel(inputShape):
    model = keras.Sequential()

    # model.add(keras.layers.LSTM(128, input_shape=inputShape, return_sequences=True,
    #                             kernel_regularizer=keras.regularizers.l2(0.001)))
    # model.add(keras.layers.LSTM(128, return_sequences=True,
    #                             kernel_regularizer=keras.regularizers.l2(0.001)))
    # model.add(keras.layers.Reshape((inputShape[0], 128, 1)))

    model.add(keras.layers.Conv2D(32, (3, 2), activation="relu", input_shape=inputShape,
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(48, (3, 2), activation="relu", input_shape=inputShape,
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding="same"))
    #####################################################################

    model.add(keras.layers.Conv2D(64, (3, 2), activation="relu", input_shape=inputShape,
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(80, (3, 2), activation="relu", input_shape=inputShape,
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding="same"))

    #####################################################################
    model.add(keras.layers.Conv2D(128, (3, 2), activation="relu", input_shape=inputShape,
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(256, (3, 2), activation="relu", input_shape=inputShape,
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding="same"))
    #####################################################################

    # output
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
    return np.array([dataset["data"][segRange[0]:segRange[1]] for segRange in batch])


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
        # for i in range(batchSize):
        valIdcs = np.random.choice(vX.shape[0], int(vX.shape[0]/2))
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
    # dataset, inputShape = loadDataset(0.2)
    # model = makeModel(inputShape)
    # plot_history(fit(model, dataset, 32, 60))
    # model.save("lastModel")

    model = keras.models.load_model("lastModel")
    for fileName in TO_OPTIMIZE_DIR.glob("*"):
        if not Path(fileName).is_file():
            continue
        mfcc = GenerateDataset.calcMFCCs(GenerateDataset.loadSignal(fileName))
        lastMFCCIdx = len(mfcc) - GenerateDataset.SEG_LEN
        stateStartIdx = 0
        prevWasGood = -1
        segmentsToCut = []

        # for i in range(0, lastMFCCIdx, GenerateDataset.SEQ_LEN):
        #     segMFCC = (mfcc[i:i+GenerateDataset.SEG_LEN])[np.newaxis, ..., np.newaxis]
        #     idx = (i+GenerateDataset.SEQ_PADDING)*GenerateDataset.HOP_LENGTH
        #     endIdx = idx + GenerateDataset.SEQ_LEN * GenerateDataset.HOP_LENGTH
        #     isGood = (model.predict(segMFCC) >= 0.3) if lastMFCCIdx - i > GenerateDataset.SEQ_LEN else (not prevWasGood)
        #     prevWasGood = prevWasGood if prevWasGood != -1 else isGood  # same as a first checked sequence
        #     if isGood and not prevWasGood:  # ending a bad state
        #         stateStartIdx = idx  # starting a good state
        #     elif not isGood and prevWasGood:  # ending a good state
        #         segmentsToCut.append((stateStartIdx / GenerateDataset.SAMPLE_RATE,
        #                               idx / GenerateDataset.SAMPLE_RATE))
        #         stateStartIdx = idx  # starting a bad state
        #     prevWasGood = isGood
        #     print("{:.2f} %\r".format(i/lastMFCCIdx * 100), end="", flush=True)

        # with open("ranges.out", "wb") as outf:
        #     pickle.dump(segmentsToCut, outf)
        with open("ranges.out", "rb") as outf:
            segmentsToCut = pickle.load(outf)

        ffmpegSelectArgs = "\'" + "+".join(["between(t,{:.3f},{:.3f})".format(seg[0], seg[1]) for seg in segmentsToCut]) + "\'"
        videoFilter = f"select={ffmpegSelectArgs},setpts=N/FRAME_RATE/TB*(1.0 - 0.0016),fps=fps=15"
        audioFilter = f"aselect={ffmpegSelectArgs},asetpts=N/SR/TB"

        if not Path(".temp/").exists():
            Path(".temp/").mkdir()
        with open(".temp/vf", "w") as vff:
            vff.write(videoFilter)
        with open(".temp/af", "w") as aff:
            aff.write(audioFilter)

        p = subprocess.Popen([f"ffmpeg -y -i {str(fileName)} -c:v h264_nvenc -c:a aac -filter_script:v .temp/vf -filter_script:a .temp/af " +
                              f"optimized/{fileName.stem}_optimized.mp4"], shell=True)
    p.wait()
    shutil.rmtree(".temp/")
