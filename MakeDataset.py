import argparse
import logging
import shutil
import youtube_dl
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count

# logging.getLogger().addHandler(logging.StreamHandler())
logger = logging.getLogger(__name__)

SUBTITLES_FORMAT = 'vtt'


def _progressCallback(stream, chunk, bytesRemaining):
    barLength = 20
    percent = (1 - bytesRemaining/stream.filesize) * 100
    arrow = '-' * int(percent/100 * barLength - 1) + '>'
    arrow += ' ' * (barLength - len(arrow))
    print(f'\rProgress: [{arrow}] {percent:.2f} % ', end='')


def _completeCallback(stream, filepath):
    print('Done')


def _inspectVidInfo(vidInfo, lang):
    title = vidInfo['title']
    subs = {sub['ext'] for sub in vidInfo['subtitles'].setdefault(lang, [])}
    if SUBTITLES_FORMAT in subs:
        return vidInfo['webpage_url']
    logger.warning(f'Skipping {title} - No subtitles for language: "{lang}" '
                   f'in format: "{SUBTITLES_FORMAT}"')
    return {}


def _inspectURL(ydl, url, lang):
    try:
        info = ydl.extract_info(url, download=False)
        if 'entries' in info:
            playlist = []
            for vidInfo in info['entries']:
                playlist.append(_inspectVidInfo(vidInfo, lang))
            return playlist
        else:
            return [_inspectVidInfo(info, lang)]
    except Exception as e:
        logger.warning(f'Skipping source: {url} - {str(e)}')
    return {}


def _getURLs(srcFile, lang, jobs):
    urls = []
    with youtube_dl.YoutubeDL() as ydl:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            lines = [url.strip() for url in srcFile.readlines()]
            futures = {executor.submit(_inspectURL, ydl, url, lang) for url in lines if url != ''}
            for future in futures:
                urls += future.result()
    return list(set(urls))


if __name__ == '__main__':
    DEFAULT_OUT_DIR = str((Path.cwd()/'data'/'eval').resolve())
    DEFAULT_SRC_FILE = str((Path.cwd()/'data'/'eval.txt').resolve())

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--src',
                        help=f'Path to the file with sources to download',
                        type=str, action='store', default=DEFAULT_SRC_FILE)
    parser.add_argument('-o', '--out',
                        help=f'Path to the directory to download to',
                        type=str, action='store', default=DEFAULT_OUT_DIR)
    parser.add_argument('-l', '--lang',
                        help=f'Language used in the videos',
                        type=str, action='store', default='en')
    parser.add_argument('-j', '--jobs',
                        help=f'Number of threads to use for downloading',
                        type=str, action='store', default=4)
    parser.add_argument('-a', '--audio',
                        help=f'Download only audio',
                        action='store_true', default=False)
    args = parser.parse_args()

    srcPath = Path(args.src)
    outPath = Path(args.out)
    lang = args.lang
    jobs = int(args.jobs)
    audioOnly = args.audio

    assert srcPath.is_file(), f'Bad sources file path: {str(srcPath)}'

    with open(srcPath, 'r') as srcFile:
        urls = _getURLs(srcFile, lang, jobs)

    def download(url, retries=3):
        try:
            options = {'format': ('worstvideo,' if not audioOnly else '') + 'worstaudio',
                       'outtmpl': f'{str(outPath)}/%(title)s-%(id)s/data.%(ext)s',
                       'restrictfilenames': 'True',
                       'writesubtitles': 'True',
                       'subtitleslangs': [lang],
                       'subtitlesformat': SUBTITLES_FORMAT}
            with youtube_dl.YoutubeDL(options) as ydl:
                ydl.download([url])
        except Exception as e:
            if retries > 0:
                download(url, retries-1)
            else:
                logger.warning(f'Download error for url: {url}')

    with Pool(jobs) as pool:
        pool.map(download, urls)
