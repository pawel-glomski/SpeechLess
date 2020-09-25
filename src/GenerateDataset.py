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
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512

SAMPLE_SEQ_LEN = int(0.2 * SAMPLE_RATE)
N_MFCCS_SEQ = int(np.ceil(SAMPLE_SEQ_LEN / HOP_LENGTH))
N_SEQ_IN_SEG = 10
SEGMENT_LEN = N_SEQ_IN_SEG * SAMPLE_SEQ_LEN


def calcMFCCs(signal):
    return librosa.feature.mfcc(signal,
                                sr=SAMPLE_RATE,
                                n_mfcc=N_MFCC,
                                n_fft=N_FFT,
                                hop_length=HOP_LENGTH).T


def extractSegments(signal, beg, end, dataset, label):
    # each "not word" sequence is filled with previous samples (or 0s if not enough), to reach the length of SEGMENT_LEN

    for begIdx in range(beg, end, SAMPLE_SEQ_LEN):
        combinedMFCCs = np.zeros(shape=(N_SEQ_IN_SEG, N_MFCCS_SEQ, N_MFCC))

        # labeled part
        tempEndIdx = min(begIdx + SAMPLE_SEQ_LEN, end)
        labeledLen = tempEndIdx - begIdx
        if labeledLen < SAMPLE_SEQ_LEN / 2:
            break
        labeledPart = np.zeros(shape=(SAMPLE_SEQ_LEN))
        labeledPart[:labeledLen] = signal[begIdx:tempEndIdx]
        combinedMFCCs[-1] = calcMFCCs(np.array(labeledPart))

        # prevPart
        for prevIdx in range(1, N_SEQ_IN_SEG):
            seqIdx = N_SEQ_IN_SEG - 1 - prevIdx
            prevBegIdx = max(begIdx - SAMPLE_SEQ_LEN * prevIdx, 0)
            prevEndIdx = begIdx - SAMPLE_SEQ_LEN * (prevIdx - 1)
            prevLen = prevEndIdx - prevBegIdx
            prevSignal = np.zeros(shape=(SAMPLE_SEQ_LEN))
            if prevLen > 0:
                prevSignal[-prevLen:] = signal[prevBegIdx:prevEndIdx]
            combinedMFCCs[seqIdx] = calcMFCCs(prevSignal)

        dataset["label"].append(label)
        dataset["MFCCs"].append(combinedMFCCs)


def generateDataset():
    if not DATA_DIR.exists():
        print("No data directory found")
        return
    dataset = {"MFCCs": [],
               "label": []}
    for fileName in DATA_DIR.glob("*.labels"):
        audioFileName = list(DATA_DIR.glob(Path(fileName).stem + ".[!labels]*"))[0]
        out, _ = (ffmpeg
                  .input(audioFileName)
                  .output('-', format='s16le', acodec='pcm_s16le', ac=1, ar=SAMPLE_RATE)
                  .overwrite_output()
                  .run(capture_stdout=True)
                  )
        signal = np.frombuffer(out, np.int16) / np.iinfo(np.int16).max

        with open(fileName) as fi:
            lastEnd = 0
            for label in list(csv.reader(fi, delimiter="\t")):
                beg = int(np.ceil(float(label[0]) * SAMPLE_RATE))
                end = min(int(float(label[1]) * SAMPLE_RATE), len(signal))  # in case audio file was trimmed
                tag = label[2]

                if tag == "n":  # not words
                    extractSegments(signal, beg, end, dataset, 0)
                    extractSegments(signal, lastEnd, beg, dataset, 1)
                lastEnd = end
            extractSegments(signal, lastEnd, len(signal), dataset, 1)

    with open(DATASET_PATH, "wb") as dsFile:
        pickle.dump(dataset, dsFile)


if __name__ == "__main__":
    generateDataset()
