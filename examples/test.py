import json
import logging
import ffmpeg
import librosa
import numpy as np
from speechless import Editor, Range
from speechless.utils import rangesOfTruth

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


def optimizeSignal(sig):
    minTrueSeq = 6
    minFalseSeq = 4

    sig = sig[:int(len(sig) / SEG_LEN) * SEG_LEN].reshape((-1, SEG_LEN))
    indicator = calcIndicator(sig.reshape(-1))
    labels = indicator >= ((np.median(indicator) + np.mean(indicator)) / 2 * (1/THRESHOLD))

    ##################### Reduce aggresive cuts #####################

    ranges = rangesOfTruth(labels == False)
    ranges = ranges[(ranges[:, 1] - ranges[:, 0]) >= minFalseSeq]
    labels[:] = True
    for r in ranges:
        labels[r[0]:r[1]] = False

    ##################### Cut out small segments #####################

    ranges = rangesOfTruth(labels)
    ranges = ranges[(ranges[:, 1] - ranges[:, 0]) >= minTrueSeq]
    labels[:] = False
    for r in ranges:
        labels[r[0]:r[1]] = True

    ranges = rangesOfTruth(labels == False)
    return (sig[labels == True].reshape(-1), ranges * SEG_LEN / SAMPLE_RATE)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        handlers=[logging.StreamHandler()],
                        format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger()

    sig = loadSignal('video.mp4')
    audio, ranges = optimizeSignal(sig)
    ratio = 0.0
    expectedDuration = (len(sig)/SAMPLE_RATE - (1-ratio)*np.sum(ranges[:, 1] - ranges[:, 0]))/60
    print('Expected duration: '
          f'{int(expectedDuration)}:{(expectedDuration - int(expectedDuration))*60}')

    ranges = Range.fromNumpy(np.concatenate([ranges, np.ones((ranges.shape[0], 1))*ratio], 1))

    # with open('examples/test.json', 'r') as fp:
    #     jsonSpecs = json.load(fp)
    #     editor = Editor.fromJSON(jsonSpecs, logger=logger)
    #     ranges = Editor.parseJSONRanges(jsonSpecs['ranges'])
    editor = Editor(logger)
    editor.specs.setdefault('audio', {})['codec'] = 'aac'
    editor.specs.setdefault('video', {})['resolution'] = [256, 144]

    editor.exportJSON(ranges, 'out.json')
    editor.edit('video.mp4', ranges, 'out.mp4')
