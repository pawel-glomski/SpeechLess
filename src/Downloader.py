import argparse
import logging
import youtube_dl
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)
nullLogger = logging.getLogger('null')
nullLogger.setLevel(logging.FATAL)


class NullLogger(object):
    def write(self, text):
        pass


SUBTITLES_FORMAT = 'vtt'
QUERY_JOBS_MULTI = 4
DOWNLOAD_RETRIES = 6
SLOW_DOWNLOAD_TIMEOUT = 3
OUT_AUD = 'audio'
OUT_VID = 'video'
OUT_SUB = 'subs'


def downloadDataset(args: object) -> type(None):
    """Download the specified dataset

    :param args: This module's possible args (defined in __main__ section of this file)
    :type args: object
    """
    assert args.src.is_file(), f'Bad src file path: {args.src}'

    print(f'Checking the links provided in: {args.src}')
    urls = _getURLs(args.src, args.lang, args.jobs * QUERY_JOBS_MULTI)
    print(f'Finished checking links')

    print('Downloading:')
    with Pool(args.jobs) as pool:
        urls = pool.map(_downloadStream, [(url, OUT_SUB, args) for url in urls])
        urls = set(urls) - {''}
        urls = pool.map(_downloadStream, [(url, OUT_AUD, args) for url in urls])
        urls = set(urls) - {''}
        if args.video:
            urls = pool.map(_downloadStream, [(url, OUT_VID, args) for url in urls])
            urls = set(urls) - {''}
    print('Finished downloading')

    with open(args.dst/'downloaded.txt', 'w') as file:
        file.writelines([url+'\n' for url in urls])


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
        with Pool(jobs) as pool:
            lines = [url.strip() for url in srcFile.readlines()]
            urlsList = pool.map(_inspectURL, [(url, lang) for url in lines if url != ''])
            urls += [url for inspURLs in urlsList for url in inspURLs]
        return list(set(urls))


def _inspectURL(url_lang: Tuple[str, str]) -> List[str]:
    """Inspect the provided URL, expand if it is a playlist, and return the URLs which have
    subtitles in the specified language

    :param url_lang: URL to inspect and language of subtitles
    :type url_lang: Tuple[str, str]
    :return: List of valid URLs
    :rtype: List[str]
    """
    url, lang = url_lang
    try:
        with youtube_dl.YoutubeDL({'logger': nullLogger}) as ydl:
            info = ydl.extract_info(url, download=False)
            if 'entries' in info:
                playlist = []
                for vidInfo in info['entries']:
                    playlist += _getValidURL(vidInfo, lang)
                return playlist
            else:
                return _getValidURL(info, lang)
    except Exception as e:
        logger.warning(f'[ERROR] {url} - {str(e)}')
    return []


def _getValidURL(vidInfo: dict, lang: str) -> List[str]:
    """Check whether the provided video match the requirements

    :param vidInfo: Information about the video
    :type vidInfo: dict
    :param lang: Required language of subtitles
    :type lang: str
    :return: List with the URL of the video or empty if it was discarded
    :rtype: List[str]
    """
    title = vidInfo['title']
    url = vidInfo['webpage_url']
    subs = {sub['ext'] for sub in vidInfo.get('subtitles', {}).setdefault(lang, [])}
    if SUBTITLES_FORMAT in subs:
        print(f'[VALID] {url}')
        return [url]
    logger.warning(f'[ BAD ] {url} - No subtitles for language: "{lang}" '
                   f'in format: "{SUBTITLES_FORMAT}"')
    return []


def _downloadStream(url_args_type: Tuple[str, str, object]) -> str:
    """Download a stream of the specified type from the provided URL

    :param url_args_type: URL, type of stream to download, and this module's args (defined in
    __main__ section of this file)
    :type url_args_type: Tuple[str, str, object]
    :raises ConnectionError: When download is going too slow, restart the connection
    :return: The provided URL if successfully downloaded, empty string otherwise
    :rtype: str
    """
    url, stype, args = url_args_type
    lastGoodSpeedTime = 0

    def _progressCallback(progress):
        nonlocal lastGoodSpeedTime
        if progress['status'] == 'downloading':
            speed = progress['speed']
            if speed > args.min:
                lastGoodSpeedTime = progress['elapsed']
            if (progress['elapsed'] - lastGoodSpeedTime > SLOW_DOWNLOAD_TIMEOUT
                    and progress['eta'] > 15):
                raise ConnectionError('Too slow ({:.2f} KiB/s), '.format(speed/1024) +
                                      'restarting the connection')

    dstPath = '{}/%(title)s-%(id)s/{}.%(ext)s'
    common_options = {'logger': nullLogger,
                      'restrictfilenames': 'True',
                      'progress_hooks': [_progressCallback],
                      'buffersize': args.buffer}

    if stype == OUT_SUB:
        if _download(url, {**common_options,
                           'outtmpl': dstPath.format(args.dst, OUT_SUB),
                           'skip_download': 'True',
                           'writesubtitles': 'True',
                           'subtitleslangs': [args.lang],
                           'subtitlesformat': SUBTITLES_FORMAT},
                     f'[!SUB!] {url} - '):
            print(f'[ SUB ] {url}')
            return url
    elif stype == OUT_AUD:
        if _download(url, {**common_options, 'outtmpl': dstPath.format(args.dst, OUT_AUD),
                           'format': 'worstaudio'},
                     f'[!AUD!] {url} - '):
            print(f'[ AUD ] {url}')
            return url
    elif stype == OUT_VID:
        assert args.video, 'Tried to download a video stream, without the --video flag'
        if _download(url, {**common_options,
                           'outtmpl': dstPath.format(args.dst, OUT_VID),
                           'format': 'worstvideo'},
                     f'[!VID!] {url} - '):
            print(f'[ VID ] {url}')
            return url
    else:
        raise ValueError(f'Bad stream type value: {stype}')
    return ''


def _download(url: str, options: dict, errPrefix: str = '') -> bool:
    """Download a stream/video specified in options with other download settings

    :param url: URL to the video containing the resource specified in options
    :type url: str
    :param options: Download options for youtube-dl downloader
    :type options: dict
    :param errPrefix: Prefix of error/warning message
    :type errPrefix: str
    :return: Whether successfully downloaded specified resource
    :rtype: bool
    """
    retries = DOWNLOAD_RETRIES
    while retries > 0:
        with youtube_dl.YoutubeDL(options) as ydl:
            try:
                ydl.download([url])
                return True
            except youtube_dl.DownloadError as e:
                logger.warning(errPrefix + str(e))
                if e.exc_info[0] is not ConnectionError:
                    retries -= 1
    logger.error(errPrefix + f'Could not download the data after {DOWNLOAD_RETRIES} retries')
    return False


if __name__ == '__main__':
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
                        type=float, action='store', default=0.1)
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
