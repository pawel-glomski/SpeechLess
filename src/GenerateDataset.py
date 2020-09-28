# %%

from pathlib import Path
import numpy as np
import ffmpeg
import librosa
import librosa.display
import matplotlib.pyplot as plt
import csv
import pickle

DATA_DIR = (Path.cwd()/("../data" if Path.cwd().name == "src" else "data")).resolve()
DATASET_PATH = DATA_DIR/"dataset"

SAMPLE_RATE = 22050
N_MFCC = 25
N_FFT = 2048
HOP_LENGTH = 512
N_MFCC_VECS_IN_SEC = int(np.ceil(SAMPLE_RATE / HOP_LENGTH))

# in MFCCs vectors
SEQ_LEN = int(np.ceil(0.15 * N_MFCC_VECS_IN_SEC))
SEQ_PADDING = int(np.ceil(1.5 * N_MFCC_VECS_IN_SEC))
SEQ_HOP_LENGTH = int(SEQ_LEN / 2)
SEG_LEN = SEQ_LEN + 2*SEQ_PADDING


def calcMFCCs(signal):
    return librosa.feature.mfcc(signal,
                                sr=SAMPLE_RATE,
                                n_mfcc=N_MFCC,
                                n_fft=N_FFT,
                                hop_length=HOP_LENGTH).T


def loadSignal(fileName):
    audioFileName = list(DATA_DIR.glob(Path(fileName).stem + ".[!labels]*"))[0]
    out, _ = (ffmpeg
              .input(audioFileName)
              .output('-', format='s16le', acodec='pcm_s16le', ac=1, ar=SAMPLE_RATE)
              .overwrite_output()
              .run(capture_stdout=True)
              )
    return np.frombuffer(out, np.int16) / np.iinfo(np.int16).max


def timeToMFCCIdx(time):
    return int(np.ceil(float(time) * SAMPLE_RATE) / HOP_LENGTH)


def generateDataset():
    if not DATA_DIR.exists():
        print("No data directory found")
        return

    dataset = {"MFCCs": [],
               "label": []}
    for fileName in DATA_DIR.glob("*.labels"):
        with open(fileName) as fi:
            # prepare mfcc and their targets
            mfcc = calcMFCCs(loadSignal(fileName))
            targets = np.ones(shape=(len(mfcc)))
            for (beg, end, tag) in csv.reader(fi, delimiter="\t"):
                targets[timeToMFCCIdx(beg): timeToMFCCIdx(end)] = tag == "r"

            # prepare segments
            for i in range(0, len(mfcc) - SEG_LEN, SEQ_HOP_LENGTH):
                dataset["MFCCs"].append(mfcc[i:i+SEG_LEN])
                dataset["label"].append((np.mean(targets[i+SEQ_PADDING:i+SEQ_PADDING+SEQ_LEN])) > 0.4)
    dataset["MFCCs"] = np.array(dataset["MFCCs"])
    dataset["label"] = np.array(dataset["label"])
    print(np.mean(dataset["label"]))
    with open(DATASET_PATH, "wb") as dsFile:
        pickle.dump(dataset, dsFile)
        print("Generated {} samples".format(len(dataset["MFCCs"])))


if __name__ == "__main__":
    generateDataset()
