import argparse
import logging
import shutil
import pytube.request
from urllib import parse
from pytube import Playlist, YouTube
from pathlib import Path

# logging.getLogger().addHandler(logging.StreamHandler())
logger = logging.getLogger(__name__)

OUT_DIR = str((Path.cwd()/'data').resolve())
SRC_FILE = str((Path(OUT_DIR)/'eval.txt').resolve())
LANG = 'en'
RES = '144p'
ABR = '50kbps'
AUDIO_ONLY = False

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--src',
                    help=f'Path to the file with sources to download, default: {SRC_FILE}',
                    type=str, action='store', default=SRC_FILE)
parser.add_argument('-o', '--out',
                    help=f'Path to the directory to download to, default: {OUT_DIR}',
                    type=str, action='store', default=OUT_DIR)
parser.add_argument('-l', '--lang',
                    help=f'Language used in the videos, default: {LANG}',
                    type=str, action='store', default=LANG)
parser.add_argument('-a', '--audio',
                    help=f'Download only audio, default: {AUDIO_ONLY}',
                    action='store_true', default=AUDIO_ONLY)
parser.add_argument('-r', '--res',
                    help=f'Resolution of videos, default: {RES}',
                    type=str, action='store', default=RES)
parser.add_argument('-b', '--br',
                    help=f'Audio bit rate, default: {ABR}',
                    type=str, action='store', default=ABR)
args = parser.parse_args()


OUT_DIR = Path(args.out)
SRC_FILE = Path(args.src)
LANG = args.lang
AUDIO_ONLY = args.audio
RES = None if AUDIO_ONLY else args.res
ABR = args.br


def _progressCallback(stream, chunk, bytesRemaining):
    barLength = 20
    percent = (1 - bytesRemaining/stream.filesize) * 100
    arrow = '-' * int(percent/100 * barLength - 1) + '>'
    arrow += ' ' * (barLength - len(arrow))
    print(f'\rProgress: [{arrow}] {percent:.2f} % ', end='')


def _completeCallback(stream, filepath):
    print('Done')


def _downloadCaptions(media, outName, outDir):
    captions = None
    if LANG in media.captions:
        captions = media.captions[LANG]
    else:
        for lang in media.captions.lang_code_index:
            if lang.startswith(LANG):
                captions = media.captions[lang]
                break
    assert captions is not None, f'No captions for language: {LANG}'

    outNameC = outName + '_c'
    print('Downloading captions...')
    captions.download(outNameC, srt=True, output_path=outDir)
    print('Done')


def _downloadAudio(media, outName, outDir):
    def abrVal(abr: str) -> int:
        return int(abr[:-len('kbps')])

    astreams = media.streams.filter(adaptive=True, only_audio=True)
    assert len(astreams) > 0, 'No audio stream'

    abrStreams = {}
    for s in astreams.fmt_streams:
        abrStreams.setdefault(s.abr, []).append(s)

    astream = None
    for i, abr in enumerate(sorted(abrStreams.keys(), key=abrVal)):
        if abrVal(abr) >= abrVal(ABR) or i + 1 == len(abrStreams):
            if abr != ABR:
                logger.warn(f'No audio stream with {ABR} bit rate, downloading {abr} instead')
            astream = sorted(abrStreams[abr], key=lambda x: x.bitrate)[0]
            break
    assert astream is not None, 'Logic error'

    outNameA = outName + '_a'
    print('Downloading audio...')
    astream.download(filename=outNameA, output_path=outDir)


def _downloadVideo(media, outName, outDir):
    if AUDIO_ONLY:
        return

    def resVal(res: str) -> int:
        return int(res[:-len('p')])

    vstreams = media.streams.filter(adaptive=True, only_video=True)
    assert len(vstreams) > 0, 'No video stream'

    resStreams = {}
    for s in vstreams.fmt_streams:
        resStreams.setdefault(s.resolution, []).append(s)

    vstream = None
    for i, res in enumerate(sorted(resStreams.keys(), key=resVal)):
        if resVal(res) >= resVal(RES) or i + 1 == len(resStreams):
            if res != RES:
                logger.warn(f'No video stream with {RES} resolution, downloading {res} instead')
            vstream = sorted(resStreams[res], key=lambda x: x.bitrate)[0]
            break
    assert vstream is not None, 'Logic error'

    outNameV = outName + '_v'
    print('Downloading video...')
    vstream.download(filename=outNameV, output_path=outDir)


def downloadFromYouTube(url, outDir):
    try:
        media = YouTube(url, on_progress_callback=_progressCallback,
                        on_complete_callback=_completeCallback)
        media.check_availability()
        outName = media.title.replace(' ', '_')

        print(f'Getting {outName}:')
        _downloadCaptions(media, outName, outDir)
        _downloadAudio(media, outName, outDir)
        _downloadVideo(media, outName, outDir)
    except Exception as e:
        logger.warn(f'Skipping source: {url} - {str(e)}')
        shutil.rmtree(outDir)


def makeDataset():
    assert OUT_DIR.is_dir(), f'Bad output directory path: {str(SRC_FILE)}'
    assert SRC_FILE.is_file() and SRC_FILE.exists(), f'Bad sources file path: {str(OUT_DIR)}'

    if not OUT_DIR.exists():
        OUT_DIR.mkdir()

    with open(SRC_FILE, 'r') as srcFile:
        for i, url in enumerate(srcFile.readlines()):
            url = url.strip()
            try:
                pURL = parse.urlparse(url)
                assert pURL.hostname == 'www.youtube.com', 'wrong hostname'

                outDir = OUT_DIR/f'{i+1}'
                if not outDir.exists():
                    outDir.mkdir()

                if pURL.path == '/watch':
                    downloadFromYouTube(url, outDir)
                elif pURL.path == '/playlist':
                    pl = Playlist(url)
                    for j, vidURL in enumerate(pl.video_urls):
                        downloadFromYouTube(vidURL, outDir/f'{j+1}')
            except Exception as e:
                logger.warn(f'Skipping source: {url} - {str(e)}')


if __name__ == '__main__':
    makeDataset()
