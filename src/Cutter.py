from sklearn.model_selection import train_test_split
from scipy.interpolate import CubicSpline
from pathlib import Path

import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import GenerateDataset as gd

import subprocess
import pickle
import shutil

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

TO_OPTIMIZE_DIR = (Path.cwd()/("../toOptimize" if Path.cwd().name == "src" else "toOptimize")).resolve()
V_FPS = 15
THRESHOLD = 0.77

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def loadDataset(validationRatio):
    dataset = pickle.load(dsFile)
    dataset["data"] = (dataset["data"].astype(np.float32) / np.iinfo(dataset["data"].dtype).max)[..., np.newaxis]
    temp = dataset["range"][0]
    return (dataset, (temp[1] - temp[0], *dataset["data"].shape[1:]), dataset["label"][0].shape)


def makeModel(inputShape, outputShape):
    # xIn = keras.Input(shape=inputShape)
    # x = layers.BatchNormalization()(xIn)
    # x = layers.Conv1D(256, kernel_size=17, strides=8, activation="relu", padding="same", kernel_regularizer=tf.keras.regularizers.L2(0.001))(x)
    # xDil1 = layers.Conv1D(32, kernel_size=3, dilation_rate=1, activation="relu",
    #                       padding="same", kernel_regularizer=tf.keras.regularizers.L2(0.001))(x)
    # xDil2 = layers.Conv1D(32, kernel_size=3, dilation_rate=3, activation="relu", padding="same",
    #                       kernel_regularizer=tf.keras.regularizers.L2(0.001))(xDil1)
    # xDil3 = layers.Conv1D(32, kernel_size=3, dilation_rate=9, activation="relu", padding="same",
    #                       kernel_regularizer=tf.keras.regularizers.L2(0.001))(xDil2)
    # xDil4 = layers.Conv1D(32, kernel_size=3, dilation_rate=27, activation="relu", padding="same",
    #                       kernel_regularizer=tf.keras.regularizers.L2(0.001))(xDil3)
    # x = layers.Concatenate()([xDil1, xDil2, xDil3, xDil4, x])
    # x = layers.BatchNormalization()(x)

    # x = layers.Conv1D(256, kernel_size=9, strides=4, activation="relu", padding="same", kernel_regularizer=tf.keras.regularizers.L2(0.001))(x)
    # xDil1 = layers.Conv1D(32, kernel_size=3, dilation_rate=1, activation="relu",
    #                       padding="same", kernel_regularizer=tf.keras.regularizers.L2(0.001))(x)
    # xDil2 = layers.Conv1D(32, kernel_size=3, dilation_rate=3, activation="relu", padding="same",
    #                       kernel_regularizer=tf.keras.regularizers.L2(0.001))(xDil1)
    # xDil3 = layers.Conv1D(32, kernel_size=3, dilation_rate=9, activation="relu", padding="same",
    #                       kernel_regularizer=tf.keras.regularizers.L2(0.001))(xDil2)
    # xDil4 = layers.Conv1D(32, kernel_size=3, dilation_rate=27, activation="relu", padding="same",
    #                       kernel_regularizer=tf.keras.regularizers.L2(0.001))(xDil3)
    # x = layers.Concatenate()([xDil1, xDil2, xDil3, xDil4, x])
    # x = layers.BatchNormalization()(x)

    # x = layers.Conv1D(256, kernel_size=5, strides=2, activation="relu", padding="same", kernel_regularizer=tf.keras.regularizers.L2(0.001))(x)
    # xDil1 = layers.Conv1D(32, kernel_size=3, dilation_rate=1, activation="relu",
    #                       padding="same", kernel_regularizer=tf.keras.regularizers.L2(0.001))(x)
    # xDil2 = layers.Conv1D(32, kernel_size=3, dilation_rate=3, activation="relu", padding="same",
    #                       kernel_regularizer=tf.keras.regularizers.L2(0.001))(xDil1)
    # xDil3 = layers.Conv1D(32, kernel_size=3, dilation_rate=9, activation="relu", padding="same",
    #                       kernel_regularizer=tf.keras.regularizers.L2(0.001))(xDil2)
    # xDil4 = layers.Conv1D(32, kernel_size=3, dilation_rate=27, activation="relu", padding="same",
    #                       kernel_regularizer=tf.keras.regularizers.L2(0.001))(xDil3)
    # x = layers.Concatenate()([xDil1, xDil2, xDil3, xDil4, x])
    # x = layers.BatchNormalization()(x)

    # x = layers.Conv1D(256, kernel_size=5, strides=2, activation="relu", padding="same", kernel_regularizer=tf.keras.regularizers.L2(0.001))(x)
    # x = layers.Conv1D(256, kernel_size=3, strides=2, activation="relu", padding="same", kernel_regularizer=tf.keras.regularizers.L2(0.001))(x)
    # x = layers.Conv1D(512, kernel_size=3, strides=2, activation="relu", padding="same", kernel_regularizer=tf.keras.regularizers.L2(0.001))(x)
    # x = layers.Flatten()(x)
    # x = layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.L2(0.001))(x)
    # x = layers.Dropout(0.2)(x)
    # xOut = layers.Dense(outputShape[0], activation="sigmoid")(x)

    # model = keras.Model(inputs=xIn, outputs=xOut)
    # model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    #               loss='binary_crossentropy',
    #               metrics=['accuracy'])
    model = keras.models.load_model("lastModel")
    model.summary()
    return model


def splitDataset(dataset, validationPart):
    return train_test_split(dataset["range"], dataset["label"], test_size=validationPart)


def prepareBatch(batch, dataset):
    return np.array([dataset["data"][segRange[0]:segRange[1]] for segRange in batch])


def fit(model, dataset, batchSize, epochs, validationPart=0.2):
    x, vX, y, vY = splitDataset(dataset, validationPart)

    epochLen = int(x.shape[0] / batchSize)
    validLen = int(vX.shape[0] / batchSize / 2)
    for e in range(epochs):
        # train
        for i, trainIdcs in enumerate(np.random.permutation(epochLen * batchSize).reshape(epochLen, batchSize)):
            trainX = prepareBatch(np.take(x, trainIdcs, axis=0), dataset)
            trainY = np.take(y, trainIdcs, axis=0)
            trLoss, trAcc = model.train_on_batch(trainX, trainY, reset_metrics=False)
            print(f"Epoch #{e+1}: {i / epochLen * 100:.2f}%: trAcc = {trAcc:.4f}, trLoss = {trLoss:.4f}\r", end="", flush=True)
        model.reset_metrics()
        # validate
        for i, valIdcs in enumerate(np.random.permutation(validLen * batchSize).reshape(validLen, batchSize)):
            valX = prepareBatch(np.take(vX, valIdcs, axis=0), dataset)
            valY = np.take(vY, valIdcs, axis=0)
            vaLoss, vaAcc = model.test_on_batch(valX, valY, reset_metrics=False)
        model.reset_metrics()

        print(f"Epoch #{e+1}: Done! trAcc = {trAcc:.4f}, trLoss = {trLoss:.4f}, vaAcc = {vaAcc:.4f}, vaLoss = {vaLoss:.4f}")

        if e % 5 == 4:
            model.save("lastModel")


with open(gd.DATASET_PATH, "rb") as dsFile:
    dataset, inShape, outShape = loadDataset(0.2)
    model = makeModel(inShape, outShape)
    fit(model, dataset, 32, 60)
    model.save("lastModel")

    model = keras.models.load_model("lastModel")
    for fileName in TO_OPTIMIZE_DIR.glob("*"):
        if not Path(fileName).is_file():
            continue
        sig = gd.loadSignal(fileName)
        sig = sig[:int(len(sig) / gd.SEG_LEN) * gd.SEG_LEN] / np.iinfo(sig.dtype).max

        labels = np.zeros(int(len(sig) / gd.SEG_CELL_LEN))
        for i, offset in enumerate(range(0, gd.SEG_LEN, int(gd.SEG_LEN/4))):
            tempSig = np.concatenate([sig[offset:], np.zeros(offset)]).reshape((-1, gd.SEG_LEN, 1))
            cellIdx = int(offset/gd.SEG_CELL_LEN)
            labels[cellIdx:] *= i
            labels[cellIdx:] += model.predict(tempSig, verbose=True).reshape(-1)[:len(labels)-cellIdx]
            labels[cellIdx:] /= i+1
        labels = labels >= THRESHOLD

        # with open("labels.save", "wb") as lf:
        #     pickle.dump(labels, lf)
        # with open("labels.save", "rb") as lf:
        #     labels = pickle.load(lf)

        # for tresh in np.linspace(0.2, 0.9, 30):
        #     tmp = labels >= tresh
        #     for i in range(1, len(tmp)-1):
        #         tmp[i] = (tmp[i-1] and tmp[i+1]) or (tmp[i] and (tmp[i-1] or tmp[i+1]))  # remove singular (alone) cells
        #     audio = sig.reshape((-1, gd.SEG_CELL_LEN))[tmp].reshape(-1)
        #     sf.write(f"test_{tresh:.2f}.wav", audio, gd.SAMPLE_RATE)

        for i in range(1, len(labels)-1):
            labels[i] = (labels[i-1] and labels[i+1]) or (labels[i] and (labels[i-1] or labels[i+1]))  # remove singular (alone) cells

        ranges = np.where(labels[:-1] != labels[1:])[0] + 1
        isEven = len(ranges) % 2 == 0
        if (isEven and labels[0]) or (not isEven and labels[0]):
            ranges = np.concatenate([[0], ranges])
        if (isEven and labels[0]) or (not isEven and not labels[0]):
            ranges = np.concatenate([ranges, [len(labels)]])
        ranges = ranges.reshape((-1, 2))

        framesToGet = []
        videoLen = 0
        audioLen = 0
        for r in ranges * (gd.SEG_CELL_LEN / gd.SAMPLE_RATE):
            sF = np.floor(r[0] * V_FPS)
            eF = np.ceil(r[1] * V_FPS)
            audioLen += r[1] - r[0]
            framesToAdd = np.min([np.round((audioLen - videoLen)*V_FPS), eF - sF])
            if framesToAdd > 0:
                videoLen += framesToAdd / V_FPS
                if len(framesToGet) > 0 and framesToGet[-1][1] + 1 >= sF:
                    framesToGet[-1][1] += framesToAdd
                else:
                    framesToGet.append([sF, sF+framesToAdd-1])
        framesToGet = np.array(framesToGet, dtype=np.int32)

        framesSelect = "\'" + "+".join([f"between(n,{seg[0]},{seg[1]})" for seg in framesToGet]) + "\'"
        videoFilter = f"select={framesSelect},setpts=N/FR/TB"

        if not Path(".temp/").exists():
            Path(".temp/").mkdir()
        with open(".temp/vf", "w") as vff:
            vff.write(videoFilter)

        p = subprocess.Popen([f"ffmpeg -y -i {str(fileName)} -an -vsync cfr -r {V_FPS} " +
                              f"-c:v hevc_nvenc -preset fast .temp/norm.mp4"], shell=True)

        audio = sig.reshape((-1, gd.SEG_CELL_LEN))[labels].reshape(-1)
        sf.write(f".temp/audio.wav", audio, gd.SAMPLE_RATE)
        p.wait()

        subprocess.Popen([f"ffmpeg -y -i .temp/norm.mp4 -i .temp/audio.wav -c:v libx265 -preset ultrafast -filter_script:v .temp/vf " +
                          f"-c:a aac -map 0:v:0 -map 1:a:0 -vsync cfr -r {V_FPS} optimized/{fileName.stem}_{THRESHOLD}_optimized.mp4"], shell=True).wait()

    shutil.rmtree(".temp/")
