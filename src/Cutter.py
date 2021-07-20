from pathlib import Path

import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import ffmpeg
import librosa

import subprocess
import shutil

TO_OPTIMIZE_DIR = (Path.cwd()/("../toOptimize" if Path.cwd().name ==
                   "src" else "toOptimize")).resolve()
V_FPS = 15
THRESHOLD = 1.5

SAMPLE_RATE = 22050
SEG_LEN = 512
MEL_HOP = SEG_LEN
MEL_N_FFT = int(2*MEL_HOP)


def loadSignal(fileName):
    out, _ = (ffmpeg
              .input(fileName)
              .output('-', format='f32le', acodec='pcm_f32le', ac=1, ar=SAMPLE_RATE)
              .overwrite_output()
              .run(capture_stdout=True))
    out = np.frombuffer(out, np.float32)
    return out / max(np.max(out), abs(np.min(out)))


def calcIndicator(sig):
    # it generates one more for some reason
    y = librosa.power_to_db(librosa.feature.melspectrogram(
        sig, sr=SAMPLE_RATE, n_fft=MEL_N_FFT, hop_length=MEL_HOP).T)[:-1]
    y -= np.median(y)
    y[y < 0] = 0
    y /= np.max(y)
    y1 = (y[1:, :]-y[:-1, :])**2
    y[:] = 0
    y[:-1, :] = y1
    y[1:, :] += y1
    y[1:-1, :] /= 2
    y = np.mean(y, axis=1)
    N = int(SAMPLE_RATE/MEL_HOP/20)
    n = y.shape[0]
    y2 = y
    for i in range(2*N+1):
        if i != N:
            y2[N:-N] += y[i:n-2*N+i]/(2*N+1)
    return y2


def trueRanges(arr):
    ranges = np.where(arr[:-1] != arr[1:])[0] + 1
    isEven = len(ranges) % 2 == 0
    if (isEven and arr[0]) or (not isEven and arr[0]):
        ranges = np.concatenate([[0], ranges])
    if (isEven and arr[0]) or (not isEven and not arr[0]):
        ranges = np.concatenate([ranges, [len(arr)]])
    return ranges.reshape((-1, 2))


def optimizeSignal(sig):
    minTrueSeq = 6
    minFalseSeq = 4

    sig = sig[:int(len(sig) / SEG_LEN) * SEG_LEN].reshape((-1, SEG_LEN))
    indicator = calcIndicator(sig.reshape(-1))
    labels = indicator >= ((np.median(indicator) + np.mean(indicator)) / 2 * (1/THRESHOLD))

    ##################### Reduce aggresive cuts #####################

    ranges = trueRanges(labels == False)
    ranges = ranges[(ranges[:, 1] - ranges[:, 0]) >= minFalseSeq]
    labels[:] = True
    for r in ranges:
        labels[r[0]:r[1]] = False

    ##################### Cut out small segments #####################

    ranges = trueRanges(labels)
    ranges = ranges[(ranges[:, 1] - ranges[:, 0]) >= minTrueSeq]
    labels[:] = False
    for r in ranges:
        labels[r[0]:r[1]] = True

    ranges = trueRanges(labels == False)
    return (sig[labels].reshape(-1), ranges * (SEG_LEN / SAMPLE_RATE))


if __name__ == "__main__":
    for fileName in TO_OPTIMIZE_DIR.glob("*"):
        if not Path(fileName).is_file():
            continue

        sig = loadSignal(fileName)
        audio, ranges = optimizeSignal(sig)
        length = np.sum(ranges[:, 1] - ranges[:, 0])
        audio_len = audio.shape[0] / SAMPLE_RATE
        if not Path(".temp/").exists():
            Path(".temp/").mkdir()

        with open('.temp/ranges.txt', 'w') as file:
            file.write(';\n')  # use default out specs
            for r in ranges:
                file.write(f'{r[0]} {r[1]} 0.0\n')

        p = subprocess.Popen([f'bin/editor "{str(fileName)}" ".temp/ranges.txt" ' +
                              f'".temp/video.mp4"'], shell=True)
        sf.write(f".temp/audio.wav", audio, SAMPLE_RATE)
        p.wait()
        p = subprocess.Popen(['ffmpeg -y -i ".temp/video.mp4" -i ".temp/audio.wav" '
                              '-map 0:v:0 -map 1:a:0 -codec:v copy '
                              f'"optimized/{fileName.stem}_optimized.mp4"'], shell=True)
        p.wait()
    if Path(".temp/").exists():
        shutil.rmtree(".temp/")
