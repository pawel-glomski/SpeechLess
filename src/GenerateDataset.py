# %%

from pathlib import Path
import numpy as np
import ffmpeg
import librosa
import librosa.display
import matplotlib.pyplot as plt
import csv
import pickle
import soundfile

DATA_DIR = (Path.cwd()/("../data" if Path.cwd().name == "src" else "data")).resolve()
DATASET_PATH = DATA_DIR/"dataset"

SAMPLE_RATE = 22050
N_MFCC = 25
N_FFT = 2048
HOP_LENGTH = 512
N_MFCC_VECS_IN_SEC = int(np.ceil(SAMPLE_RATE / HOP_LENGTH))

# in MFCCs vectors
SEQ_LEN = int(np.ceil(0.15 * N_MFCC_VECS_IN_SEC))
SEQ_PADDING = int(np.ceil(1.0 * N_MFCC_VECS_IN_SEC))
SEQ_HOP_LENGTH = int(SEQ_LEN / 2)
SEG_LEN = SEQ_LEN + 2*SEQ_PADDING


def calcMFCCs(signal):
    return librosa.feature.mfcc(signal,
                                sr=SAMPLE_RATE,
                                n_mfcc=N_MFCC,
                                n_fft=N_FFT,
                                hop_length=HOP_LENGTH).T
    # return np.abs(librosa.core.stft(signal, N_FFT, HOP_LENGTH)).T
    # return librosa.feature.melspectrogram(signal,
    #                                       sr=SAMPLE_RATE,
    #                                       n_fft=N_FFT,
    #                                       hop_length=HOP_LENGTH).T


def loadSignal(fileName):
    out, _ = (ffmpeg
              .input(fileName)
              .output('-', format='s16le', acodec='pcm_s16le', ac=1, ar=SAMPLE_RATE)
              .overwrite_output()
              .run(capture_stdout=True))
    return np.frombuffer(out, np.int16) / np.iinfo(np.int16).max


def timeToMFCCIdx(time):
    return int(np.ceil(float(time) * SAMPLE_RATE) / HOP_LENGTH)


def generateDataset():
    if not DATA_DIR.exists():
        print("No data directory found")
        return
    dataset = {"data": [],
               "range": [],
               "label": []}
    idxOffset = 0

    for fileName in DATA_DIR.glob("*.labels"):
        with open(fileName) as fi:
            # prepare MFCCs and targets
            audSig = loadSignal(list(DATA_DIR.glob(Path(fileName).stem + ".[!labels]*"))[0])
            mfcc = calcMFCCs(audSig)
            targets = np.ones(mfcc.shape[0])
            for (beg, end, tag) in csv.reader(fi, delimiter="\t"):
                targets[timeToMFCCIdx(beg): timeToMFCCIdx(end)] = (tag == "r")

            # save ranges and labels of segments
            for i in range(0, mfcc.shape[0] - SEG_LEN, SEQ_HOP_LENGTH):
                label = float(np.mean(targets[i+SEQ_PADDING:i+SEQ_PADDING+SEQ_LEN]) >= 0.3)
                dataset["range"].append(np.array([idxOffset + i, idxOffset + i + SEG_LEN]))
                dataset["label"].append(label)
                if label == 1:
                    newPart = audSig[(i + SEQ_PADDING)*HOP_LENGTH:
                                     (i + SEQ_PADDING + SEQ_LEN)*HOP_LENGTH]

            # save MFCCs
            dataset["data"].append(mfcc[: i+SEG_LEN])  # include only MFCCs that are labeled
            idxOffset += len(dataset["data"][-1])

    dataset["data"] = np.concatenate(dataset["data"], axis=0)
    dataset["range"] = np.array(dataset["range"])
    dataset["label"] = np.array(dataset["label"])
    with open(DATASET_PATH, "wb") as dsFile:
        pickle.dump(dataset, dsFile)
        print("Generated {} samples. Average label = {:.4f}".format(dataset["label"].shape[0], np.mean(dataset["label"])))


if __name__ == "__main__":
    generateDataset()
