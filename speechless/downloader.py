import argparse
import youtube_dl
from multiprocessing import Pool
from logging import Logger
from pathlib import Path
from typing import List

from .utils import NULL_LOGGER

SUBTITLES_FORMAT = 'vtt'
QUERY_JOBS_MULTI = 4
DOWNLOAD_RETRIES = 6
SLOW_DOWNLOAD_TIMEOUT = 3
OUT_AUD = 'audio'
OUT_VID = 'video'
OUT_SUB = 'subs'


class Downloader:
    def __init__(self, dst: str, lang: str, jobs: int, minSpeed: float,
                 bufferSize: float, withVideo: bool, logger: Logger = NULL_LOGGER):
        self.dst = Path(dst).resolve()
        self.lang = lang
        self.jobs = jobs
        self.minSpeed = minSpeed * 1024*1024
        self.bufferSize = int(bufferSize * 1024*1024)
        self.withVideo = withVideo
        self.logger = logger

    def download(self, src: str) -> None:
        """Download the specified dataset

        Args:
            src (str): Path to the file containing links of videos to download
        """
        src = Path(src).resolve()

        if not src.is_file():
            raise FileNotFoundError(f'Bad src file path: {src}')

        self.logger.info(f'Checking the links provided in: {src}')
        urls = self._getURLs(src, self.lang, self.jobs * QUERY_JOBS_MULTI)
        self.logger.info('Finished checking links')

        self.logger.info('Downloading:')
        with Pool(self.jobs) as pool:
            urls = pool.starmap(self._downloadStream, [(url, OUT_SUB) for url in urls])
            urls = set(urls) - {''}
            urls = pool.starmap(self._downloadStream, [(url, OUT_AUD) for url in urls])
            urls = set(urls) - {''}
            if self.withVideo:
                urls = pool.starmap(self._downloadStream, [(url, OUT_VID) for url in urls])
                urls = set(urls) - {''}
        self.logger.info('Finished downloading')

        with open(self.dst/'downloaded.txt', 'w') as file:
            file.writelines([url+'\n' for url in urls])

    def _getURLs(self, srcPath: Path, lang: str, jobs: int) -> List[str]:
        """Get a list of URLs from the provided file, which have subtitles in the specified language

        Args:
            srcPath (Path): Path to a file with links of videos to download
            lang (str): Language of subtitles
            jobs (int): Number of simultaneous queries for videos

        Returns:
            List[str]: List of unique urls to download
        """
        with open(srcPath, 'r') as srcFile:
            validUrls = []
            with Pool(jobs) as pool:
                urls = [l.strip() for l in srcFile.readlines() if l.strip() != '']
                urlsList = pool.starmap(self._inspectURL, [(url, lang) for url in urls])
                validUrls += [url for inspURLs in urlsList for url in inspURLs]
            return list(set(validUrls))

    def _inspectURL(self, url: str, lang: str) -> List[str]:
        """Inspect the provided URL, expand if it is a playlist, and return the URLs which have
        subtitles in the specified language

        Args:
            url (str): URL to inspect
            lang (str): Language of subtitles

        Returns:
            List[str]: List of valid URLs
        """
        try:
            with youtube_dl.YoutubeDL({'logger': NULL_LOGGER}) as ydl:
                info = ydl.extract_info(url, download=False)
                if 'entries' in info:
                    playlist = []
                    for vidInfo in info['entries']:
                        playlist += self._getValidURL(vidInfo, lang)
                    return playlist
                else:
                    return self._getValidURL(info, lang)
        except Exception as e:
            self.logger.warning(f'[ERROR] {url} - {str(e)}')
        return []

    def _getValidURL(self, vidInfo: dict, lang: str) -> List[str]:
        """Check whether the provided video match the requirements

        Args:
            vidInfo (dict): Information about the video
            lang (str): Required language of subtitles

        Returns:
            List[str]: List with the URL of the video or empty if it was discarded
        """
        url = vidInfo['webpage_url']
        subs = {sub['ext'] for sub in vidInfo.get('subtitles', {}).setdefault(lang, [])}
        if SUBTITLES_FORMAT in subs:
            self.logger.info(f'[VALID] {url}')
            return [url]
        self.logger.warning(f'[ BAD ] {url} - No subtitles for language: "{lang}" '
                            f'in format: "{SUBTITLES_FORMAT}"')
        return []

    def _downloadStream(self, url: str, stype: str) -> str:
        """Download a stream of the specified type from the provided URL

        Args:
            url (str): URL of stream to download
            stype (str): Type of stream to download

        Raises:
            ConnectionError: When download is going too slow, restart the connection
            ValueError: 

        Returns:
            str: The provided URL if successfully downloaded, empty string otherwise
        """
        lastGoodSpeedTime = 0

        def _progressCallback(progress):
            nonlocal lastGoodSpeedTime
            if progress['status'] == 'downloading':
                speed = progress['speed']
                if speed > self.minSpeed:
                    lastGoodSpeedTime = progress['elapsed']
                if (progress['elapsed'] - lastGoodSpeedTime > SLOW_DOWNLOAD_TIMEOUT
                        and progress['eta'] > 15):
                    raise ConnectionError('Too slow ({:.2f} KiB/s), '.format(speed/1024) +
                                          'restarting the connection')

        dstPath = '{}/%(title)s-%(id)s/{}.%(ext)s'
        common_options = {'logger': NULL_LOGGER,
                          'restrictfilenames': 'True',
                          'progress_hooks': [_progressCallback],
                          'buffersize': self.bufferSize}

        if stype == OUT_SUB:
            if self._download(url, {**common_options,
                                    'outtmpl': dstPath.format(self.dst, OUT_SUB),
                                    'skip_download': 'True',
                                    'writesubtitles': 'True',
                                    'subtitleslangs': [self.lang],
                                    'subtitlesformat': SUBTITLES_FORMAT},
                              f'[ SUB ] {url} - '):
                self.logger.info(f'[ SUB ] {url}')
                return url
        elif stype == OUT_AUD:
            if self._download(url, {**common_options,
                                    'outtmpl': dstPath.format(self.dst, OUT_AUD),
                                    'format': 'worstaudio'},
                              f'[ AUD ] {url} - '):
                self.logger.info(f'[ AUD ] {url}')
                return url
        elif stype == OUT_VID:
            assert self.withVideo, 'Tried to download a video stream, without the --video flag'
            if self._download(url, {**common_options,
                                    'outtmpl': dstPath.format(self.dst, OUT_VID),
                                    'format': 'worst'},
                              f'[ VID ] {url} - '):
                self.logger.info(f'[ VID ] {url}')
                return url
        else:
            raise ValueError(f'Bad stream type value: {stype}')
        return ''

    def _download(self, url: str, options: dict, errPrefix: str = '') -> bool:
        """Download a stream/video specified in options with other download settings

        Args:
            url (str): URL to the video containing the resource specified in options
            options (dict): Download options for youtube-dl downloader
            errPrefix (str, optional): . Defaults to ''.

        Returns:
            bool: Whether successfully downloaded specified resource
        """
        retries = DOWNLOAD_RETRIES
        while retries > 0:
            with youtube_dl.YoutubeDL(options) as ydl:
                try:
                    ydl.download([url])
                    return True
                except youtube_dl.DownloadError as e:
                    self.logger.warning(errPrefix + str(e))
                    if e.exc_info[0] is not ConnectionError:
                        retries -= 1
        if retries == 0:
            self.logger.error(errPrefix + f'Skipping after {DOWNLOAD_RETRIES} retries')
        else:
            self.logger.error(errPrefix + 'Skipping')
        return False


############################################### CLI ################################################


NAME = 'downloader'
DESCRIPTION = 'Downloads specified videos'
ARG_SRC = 'src'
ARG_DST = 'dst'
ARG_LANG = 'lang'
ARG_JOBS = 'jobs'
ARG_MIN_SPEED = 'min_speed'
ARG_BUFFER_SIZE = 'buffer_size'
ARG_WITH_VIDEO = 'with_video'
DEFAULT_ARGS = {ARG_LANG: 'en',
                ARG_JOBS: 4,
                ARG_MIN_SPEED: 0.1,
                ARG_BUFFER_SIZE: 0.5,
                ARG_WITH_VIDEO: False}


def setupArgParser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Creates CLI argument parser for downloader submodule

    Returns:
        argparse.ArgumentParser: Argument parser of this submodule
    """
    parser.description = DESCRIPTION
    parser.add_argument(ARG_SRC,
                        help='Path of the file with links to videos',
                        type=str, action='store')
    parser.add_argument(ARG_DST,
                        help='Path of the destination directory for downloaded videos',
                        type=str, action='store')
    parser.add_argument('-l', f'--{ARG_LANG}',
                        help='Language used in the videos',
                        type=str, action='store', default=DEFAULT_ARGS[ARG_LANG])
    parser.add_argument('-j', f'--{ARG_JOBS}',
                        help='Number of threads to use for downloading',
                        type=int, action='store', default=DEFAULT_ARGS[ARG_JOBS])
    parser.add_argument('-m', f'--{ARG_MIN_SPEED}',
                        help='Minimum download speed to reset the connection [MiB]',
                        type=float, action='store', default=DEFAULT_ARGS[ARG_MIN_SPEED])
    parser.add_argument('-b', f'--{ARG_BUFFER_SIZE}',
                        help='Download buffer size [MiB]',
                        type=float, action='store', default=DEFAULT_ARGS[ARG_BUFFER_SIZE])
    parser.add_argument('-v', f'--{ARG_WITH_VIDEO}',
                        help='Download also video stream',
                        action='store_true', default=DEFAULT_ARGS[ARG_WITH_VIDEO])
    parser.set_defaults(run=runSubmodule)
    return parser


def runSubmodule(args: object, logger: Logger) -> None:
    """Runs this submodule

    Args:
        args (object): Arguments of this submodule (defined in Downloader.setupArgParser)
        logger (Logger): Logger
    """
    args = args.__dict__
    dl = Downloader(dst=args[ARG_DST],
                    lang=args[ARG_LANG],
                    jobs=args[ARG_JOBS],
                    minSpeed=args[ARG_MIN_SPEED],
                    bufferSize=args[ARG_BUFFER_SIZE],
                    withVideo=args[ARG_WITH_VIDEO],
                    logger=logger)
    dl.download(args[ARG_SRC])
