import argparse
import logging
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
    DESCRIPTION = 'Downloads specified videos'
    DEFAULT_ARGS = {'lang': 'en',
                    'jobs': 4,
                    'min': 0.1,
                    'buffer': 0.5,
                    'video': True}

    def __init__(self, logger: Logger):
        self.logger = logger

    def download(self, userArgs: dict) -> None:
        """Download the specified dataset

        Args:
            args (dict): This submodule's args
        """
        args = Downloader.DEFAULT_ARGS
        args.update(userArgs)
        args['src'] = args['src'].resolve()
        args['dst'] = args['dst'].resolve()
        args['min'] = args['min'] * 1024*1024
        args['buffer'] = int(args['buffer'] * 1024*1024)

        if not args['src'].is_file():
            raise FileNotFoundError(f'Bad src file path: {args["src"]}')

        self.logger.info(f'Checking the links provided in: {args["src"]}')
        urls = self._getURLs(args['src'], args['lang'], args['jobs'] * QUERY_JOBS_MULTI)
        self.logger.info(f'Finished checking links')

        self.logger.info('Downloading:')
        with Pool(args['jobs']) as pool:
            urls = pool.starmap(self._downloadStream, [(url, OUT_SUB, args) for url in urls])
            urls = set(urls) - {''}
            urls = pool.starmap(self._downloadStream, [(url, OUT_AUD, args) for url in urls])
            urls = set(urls) - {''}
            if args['video']:
                urls = pool.starmap(self._downloadStream, [(url, OUT_VID, args) for url in urls])
                urls = set(urls) - {''}
        self.logger.info('Finished downloading')

        with open(args['dst']/'downloaded.txt', 'w') as file:
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

    def _downloadStream(self, url: str, stype: str, args: object) -> str:
        """Download a stream of the specified type from the provided URL

        Args:
            url (str): URL of stream to download
            stype (str): Type of stream to download
            args (object): This module's args (defined in __main__ section of this file)

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
                if speed > args['min']:
                    lastGoodSpeedTime = progress['elapsed']
                if (progress['elapsed'] - lastGoodSpeedTime > SLOW_DOWNLOAD_TIMEOUT
                        and progress['eta'] > 15):
                    raise ConnectionError('Too slow ({:.2f} KiB/s), '.format(speed/1024) +
                                          'restarting the connection')

        dstPath = '{}/%(title)s-%(id)s/{}.%(ext)s'
        common_options = {'logger': NULL_LOGGER,
                          'restrictfilenames': 'True',
                          'progress_hooks': [_progressCallback],
                          'buffersize': args['buffer']}

        if stype == OUT_SUB:
            if self._download(url, {**common_options,
                                    'outtmpl': dstPath.format(args['dst'], OUT_SUB),
                                    'skip_download': 'True',
                                    'writesubtitles': 'True',
                                    'subtitleslangs': [args['lang']],
                                    'subtitlesformat': SUBTITLES_FORMAT},
                              f'[ SUB ] {url} - '):
                self.logger.info(f'[ SUB ] {url}')
                return url
        elif stype == OUT_AUD:
            if self._download(url, {**common_options,
                                    'outtmpl': dstPath.format(args['dst'], OUT_AUD),
                                    'format': 'worstaudio'},
                              f'[ AUD ] {url} - '):
                self.logger.info(f'[ AUD ] {url}')
                return url
        elif stype == OUT_VID:
            assert args['video'], 'Tried to download a video stream, without the --video flag'
            if self._download(url, {**common_options,
                                    'outtmpl': dstPath.format(args['dst'], OUT_VID),
                                    'format': 'worst[height>240]'},
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

    @staticmethod
    def setupArgParser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Creates CLI argument parser for downloader submodule

        Returns:
            argparse.ArgumentParser: Argument parser of this submodule
        """
        parser.description = Downloader.DESCRIPTION
        parser.add_argument('src',
                            help=f'Path to the file with links to videos',
                            type=Path, action='store')
        parser.add_argument('dst',
                            help=f'Destination directory for downloaded videos',
                            type=Path, action='store')
        parser.add_argument('-l', '--lang',
                            help=f'Language used in the videos',
                            type=str, action='store', default=Downloader.DEFAULT_ARGS['lang'])
        parser.add_argument('-j', '--jobs',
                            help=f'Number of threads to use for downloading',
                            type=int, action='store', default=Downloader.DEFAULT_ARGS['jobs'])
        parser.add_argument('-m', '--min',
                            help=f'Minimum download speed to reset the connection [MiB]',
                            type=float, action='store', default=Downloader.DEFAULT_ARGS['min'])
        parser.add_argument('-b', '--buffer',
                            help=f'Download buffer size [MiB]',
                            type=float, action='store', default=Downloader.DEFAULT_ARGS['buffer'])
        parser.add_argument('-v', '--video',
                            help=f'Download also video stream',
                            action='store_true', default=Downloader.DEFAULT_ARGS['video'])
        parser.set_defaults(run=run)
        return parser


def run(args: object, logger: Logger) -> None:
    dl = Downloader(logger)
    dl.download(args.__dict__)
