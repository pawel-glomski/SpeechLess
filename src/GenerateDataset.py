from pathlib import Path
import numpy as np
import ffmpeg
import csv
import pickle
import librosa

DATA_DIR = (Path.cwd()/("../data" if Path.cwd().name == "src" else "data")).resolve()
DATASET_PATH = DATA_DIR/"dataset"

SAMPLE_RATE = 16000
SEG_LEN = int(1.0*SAMPLE_RATE)
SEG_CELL_LEN = 128
SEG_CELLS = int(SEG_LEN / SEG_CELL_LEN) if SEG_LEN % SEG_CELL_LEN == 0 else exit()
SEG_CELLS_HOP = int(SEG_CELLS/5) if SEG_CELLS % 5 == 0 else exit()

MEL_HOP = SEG_CELL_LEN
MEL_N_FFT = 4*MEL_HOP


def loadSignal(fileName):
    out, _ = (ffmpeg
              .input(fileName)
              .output('-', format='f32le', acodec='pcm_f32le', ac=1, ar=SAMPLE_RATE)
              .overwrite_output()
              .run(capture_stdout=True))
    return np.frombuffer(out, np.float32)


def calcMelSpect(sig):
    return librosa.feature.melspectrogram(sig, sr=SAMPLE_RATE, n_fft=MEL_N_FFT, hop_length=MEL_HOP).T[:-1]  # it generates one more for some reason


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
            # prepare data and targets
            sig = loadSignal(list(DATA_DIR.glob(Path(fileName).stem + ".[!labels]*"))[0])
            sig = sig[:int(len(sig) / SEG_LEN) * SEG_LEN]
            spect = calcMelSpect(sig)
            targets = np.ones(spect.shape[0])
            for (beg, end, tag) in csv.reader(fi, delimiter="\t"):
                targets[int(float(beg) * SAMPLE_RATE / SEG_CELL_LEN):
                        int(np.ceil(float(end) * SAMPLE_RATE / SEG_CELL_LEN))] = (tag == "r")

            # save ranges and labels of segments
            for i in range(0, targets.shape[0] - SEG_CELLS, SEG_CELLS_HOP):
                dataset["range"].append(np.array([idxOffset + i, idxOffset + i + SEG_CELLS]))
                dataset["label"].append(targets[i:i+SEG_CELLS])

            # save data
            dataset["data"].append(spect)
            idxOffset += spect.shape[0]

    dataset["data"] = np.concatenate(dataset["data"], axis=0)
    dataset["range"] = np.array(dataset["range"])
    dataset["label"] = np.array(dataset["label"])
    with open(DATASET_PATH, "wb") as dsFile:
        pickle.dump(dataset, dsFile)
        print("Generated {} samples. Average label = {:.4f}".format(dataset["label"].shape[0], np.mean(dataset["label"])))

    # labels = dataset["label"].reshape(-1) == 0
    # ranges = np.where(labels[:-1] != labels[1:])[0] + 1
    # isEven = len(ranges) % 2 == 0
    # if (isEven and labels[0]) or (not isEven and labels[0]):
    #     ranges = np.concatenate([[0], ranges])
    # if (isEven and labels[0]) or (not isEven and not labels[0]):
    #     ranges = np.concatenate([ranges, [len(labels)]])
    # ranges = ranges.reshape((-1, 2))

    # with open("outLabels.labels", "w") as file:
    #     for r in ranges * SEG_CELL_LEN / SAMPLE_RATE:
    #         file.write(f"{r[0]}\t{r[1]}\n")


if __name__ == "__main__":
    generateDataset()
