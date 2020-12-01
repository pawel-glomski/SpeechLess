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
import librosa

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from GenerateDataset import SEG_CELLS

TO_OPTIMIZE_DIR = (Path.cwd()/("../toOptimize" if Path.cwd().name == "src" else "toOptimize")).resolve()
V_FPS = 15
THRESHOLD = 0.60

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def loadDataset():
    dataset = pickle.load(dsFile)
    dataset["data"] = dataset["data"][..., np.newaxis]
    temp = dataset["range"][0]
    return (dataset, (temp[1] - temp[0], *dataset["data"].shape[1:]), dataset["label"][0].shape)


def makeModel(inputShape, outputShape):
    # xIn = keras.Input(shape=inputShape)

    # x = keras.layers.Conv2D(160, kernel_size=(3, 3), strides=(2, 3), activation="elu")(xIn)
    # x = keras.layers.MaxPool2D((2, 2))(x)

    # x = keras.layers.Conv2D(256, kernel_size=(2, 2), activation="elu")(x)
    # x = keras.layers.MaxPool2D((2, 2))(x)

    # x = keras.layers.Conv2D(256+128, kernel_size=(2, 2), activation="elu")(x)
    # x = keras.layers.MaxPool2D((2, 2))(x)
    # x = keras.layers.Conv2D(512+256, kernel_size=(2, 2), activation="elu")(x)
    # x = keras.layers.MaxPool2D((2, 2))(x)
    # x = layers.Flatten()(x)
    # x = layers.Dropout(0.5)(x)

    # x = layers.Dense(150, activation="elu")(x)
    # x = layers.Dropout(0.4)(x)
    # xOut = layers.Dense(outputShape[0], activation="sigmoid")(x)

    # model = keras.Model(inputs=xIn, outputs=xOut)
    # model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    #               loss='binary_crossentropy')
    model = keras.models.load_model("lastModel")
    model.summary()
    return model


def splitDataset(dataset, validationPart):
    return train_test_split(dataset["range"], dataset["label"], test_size=validationPart)


def prepareBatch(batch, dataset):
    return np.array([dataset["data"][segRange[0]:segRange[1]] for segRange in batch])


def fit(model, dataset, batchSize, epochs, validationPart=0.1):
    x, vX, y, vY = splitDataset(dataset, validationPart)

    epochLen = int(x.shape[0] / batchSize)
    validLen = int(vX.shape[0] / batchSize / 2)
    for e in range(epochs):
        # train
        for i, trainIdcs in enumerate(np.random.permutation(epochLen * batchSize).reshape(epochLen, batchSize)):
            trainX = prepareBatch(np.take(x, trainIdcs, axis=0), dataset)
            trainY = np.take(y, trainIdcs, axis=0)
            trLoss = model.train_on_batch(trainX, trainY, reset_metrics=False)
            print(f"Epoch #{e+1}: {i / epochLen * 100:.2f}%: trLoss = {trLoss:.4f}\r", end="", flush=True)
        model.reset_metrics()
        # validate
        for i, valIdcs in enumerate(np.random.permutation(validLen * batchSize).reshape(validLen, batchSize)):
            valX = prepareBatch(np.take(vX, valIdcs, axis=0), dataset)
            valY = np.take(vY, valIdcs, axis=0)
            vaLoss = model.test_on_batch(valX, valY, reset_metrics=False)
        model.reset_metrics()

        print(f"Epoch #{e+1}: Done! trLoss = {trLoss:.4f}, vaLoss = {vaLoss:.4f}")

        if e % 5 == 4:
            model.save("lastModel")


with open(gd.DATASET_PATH, "rb") as dsFile:
    # dataset, inShape, outShape = loadDataset()
    # model = makeModel(inShape, outShape)
    # fit(model, dataset, 32, 100)
    # model.save("lastModel")

    model = keras.models.load_model("lastModel")
    for fileName in TO_OPTIMIZE_DIR.glob("*"):
        if not Path(fileName).is_file():
            continue
        sig = gd.loadSignal(fileName)
        sig = sig[:int(len(sig) / gd.SEG_LEN) * gd.SEG_LEN]
        melIn = gd.calcMelSpect(sig)

        labels = np.zeros(int(len(sig) / gd.SEG_CELL_LEN))
        for i, offset in enumerate(range(0, gd.SEG_CELLS, int(np.ceil(gd.SEG_CELLS/3)))):
            realPart = melIn[offset:]
            compPart = np.zeros((offset, melIn.shape[1]))
            tempMel = np.concatenate([realPart, compPart]).reshape((-1, gd.SEG_CELLS, melIn.shape[1], 1))
            labels[offset:] *= i
            labels[offset:] += model.predict(tempMel, verbose=True).reshape(-1)[:len(labels)-offset] >= THRESHOLD
            labels[offset:] /= i+1
        labels = labels >= 2/3

        for i in range(1, len(labels)-1):
            labels[i] = (labels[i-1] and labels[i+1]) or (labels[i] and (labels[i-1] or labels[i+1]))  # remove singular (alone) cells

        ranges = np.where(labels[:-1] != labels[1:])[0] + 1
        isEven = len(ranges) % 2 == 0
        if (isEven and labels[0]) or (not isEven and labels[0]):
            ranges = np.concatenate([[0], ranges])
        if (isEven and labels[0]) or (not isEven and not labels[0]):
            ranges = np.concatenate([ranges, [len(labels)]])
        ranges = ranges.reshape((-1, 2))

        # with open(f"optimized/{fileName.stem}.txt", "w") as file:
        #     for r in ranges * (gd.SEG_CELL_LEN / gd.SAMPLE_RATE):
        #         file.write(f"{r[0]}\t{r[1]}\n")

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
                          f"-c:a aac -map 0:v:0 -map 1:a:0 -vsync cfr -r {V_FPS} optimized/{fileName.stem}_optimized.mp4"], shell=True).wait()

    shutil.rmtree(".temp/")
