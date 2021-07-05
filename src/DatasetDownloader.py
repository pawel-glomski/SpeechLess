import argparse
import logging
import shutil
import youtube_dl
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Tuple

# logging.getLogger().addHandler(logging.StreamHandler())
logger = logging.getLogger(__name__)

SUBTITLES_FORMAT = 'vtt'
QUERY_JOBS_MULTI = 4
DOWNLOAD_RETRIES = 6
SLOW_TIMEOUT = 3


def _getURLs(srcPath: Path, lang: str, jobs: int) -> List[str]:
    """Get a list of URLs from the provided file, which have subtitles in the specified language

    :param srcPath: Path to a file with links of videos to download
    :type srcPath: Path
    :param lang: Language of subtitles
    :type lang: str
    :param jobs: Number of simultaneous queries for videos
    :type jobs: int
    :return: List of unique urls to download
    :rtype: List[str]
    """
    with open(srcPath, 'r') as srcFile:
        urls = []
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            lines = [url.strip() for url in srcFile.readlines()]
            futures = {executor.submit(_inspectURL, url, lang) for url in lines if url != ''}
            for future in futures:
                urls += future.result()
        return list(set(urls))


def _inspectURL(url: str, lang: str) -> List[str]:
    """Inspect the provided URL, expand if it is a playlist, and return the URLs which have 
    subtitles in the specified language

    :param url: URL to inspect
    :type url: str
    :param lang: Language of subtitles
    :type lang: str
    :return: List of valid URLs
    :rtype: List[str]
    """
    try:
        with youtube_dl.YoutubeDL() as ydl:
            info = ydl.extract_info(url, download=False)
            if 'entries' in info:
                playlist = []
                for vidInfo in info['entries']:
                    playlist += _getValidURL(vidInfo, lang)
                return playlist
            else:
                return _getValidURL(info, lang)
    except Exception as e:
        logger.warning(f'Skipping source: {url} - {str(e)}')
    return []


def _getValidURL(vidInfo: dict, lang: str) -> List[str]:
    """Checks whether the provided video match the requirements

    :param vidInfo: Information about the video
    :type vidInfo: dict
    :param lang: Required language of subtitles
    :type lang: str
    :return: List with the URL of the video or empty if it was discarded
    :rtype: List[str]
    """
    title = vidInfo['title']
    subs = {sub['ext'] for sub in vidInfo.get('subtitles', {}).setdefault(lang, [])}
    if SUBTITLES_FORMAT in subs:
        return [vidInfo['webpage_url']]
    logger.warning(f'Skipping {title} - No subtitles for language: "{lang}" '
                   f'in format: "{SUBTITLES_FORMAT}"')
    return []


def _download(url_args: Tuple[str, object]) -> type(None):
    """Download the video from the provided URL

    :param url_args: URL and this module's possible args (defined in __main__ section of this file)
    :type url_args: Tuple[str, object]
    :raises ConnectionError: When download is going too slow, restart the connection
    """
    url, args = url_args
    retries = DOWNLOAD_RETRIES
    lastGoodSpeedTime = 0

    def _progressCallback(progress):
        nonlocal lastGoodSpeedTime
        if progress['status'] == 'downloading':
            speed = progress['speed']
            if speed > args.min:
                lastGoodSpeedTime = progress['elapsed']
            if progress['elapsed'] - lastGoodSpeedTime > SLOW_TIMEOUT and progress['eta'] > 15:
                raise ConnectionError('Too slow ({:.2f} KiB/s), '.format(speed/1024) +
                                      f'restarting the connection for {url}')
        elif progress['status'] == 'finished':
            print(f'[-FIN-] {url}')

    options = {'format': ('worstvideo,' if args.video else '') + 'worstaudio',
               'outtmpl': f'{args.dst}/%(title)s-%(id)s/data.%(ext)s',
               'restrictfilenames': 'True',
               'writesubtitles': 'True',
               'subtitleslangs': [args.lang],
               'subtitlesformat': SUBTITLES_FORMAT,
               'buffersize': args.buffer,
               'quiet': 'True',
               'progress_hooks': [_progressCallback]}

    print(f'[START] {url}')
    while retries > 0:
        try:
            with youtube_dl.YoutubeDL(options) as ydl:
                ydl.download([url])
            return
        except ConnectionError:
            pass
        except Exception:
            retries -= 1
    logger.warning(f'[ERROR] {url}')


def downloadDataset(args: Tuple[str, object]) -> type(None):
    """Downloads the specified dataset

    :param args: This module's possible args (defined in __main__ section of this file)
    :type args: Tuple[str, object]
    """

    assert args.src.is_file(), f'Bad src file path: {args.src}'

    print(f'Checking the links provided in: {args.src}')
    urls = _getURLs(args.src, args.lang, args.jobs * QUERY_JOBS_MULTI)
    print(f'Finished checking links')

    print('Downloading videos:')
    with Pool(args.jobs) as pool:
        pool.map(_download, [(url, args) for url in urls])
    print('Finished downloading videos')


if __name__ == '__main__':
    DEFAULT_DST_DIR = (Path.cwd()/'data'/'test').resolve()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('src',
                        help=f'Path to the file with links to videos',
                        type=Path, action='store')
    parser.add_argument('dst',
                        help=f'Destination directory for downloaded videos',
                        type=Path, action='store')
    parser.add_argument('-l', '--lang',
                        help=f'Language used in the videos',
                        type=str, action='store', default='en')
    parser.add_argument('-j', '--jobs',
                        help=f'Number of threads to use for downloading',
                        type=int, action='store', default=4)
    parser.add_argument('-m', '--min',
                        help=f'Minimum download speed to reset the connection [MiB]',
                        type=float, action='store', default=0.15)
    parser.add_argument('-b', '--buffer',
                        help=f'Download buffer size [MiB]',
                        type=float, action='store', default=0.5)
    parser.add_argument('-v', '--video',
                        help=f'Download video stream',
                        action='store_true', default=False)

    args = parser.parse_args()
    args.src = args.src.resolve()
    args.dst = args.dst.resolve()
    args.min = args.min * 1024*1024
    args.buffer = int(args.buffer * 1024*1024)

    downloadDataset(args)
