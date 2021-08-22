from setuptools import find_packages, setup

setup(
    name='speechless',
    version=open('VERSION').read().strip(),
    author='Paweł Głomski, Tomasz Rusinowicz, Jan Dorniak',
    author_email='pavel.glomski@intel.com, axontom.online@gmail.com, JanDorniak99@gmail.com',
    description='A tool for automated audio/video editing with speech processing capabilities',
    long_description=open('README.md', 'r', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    license='',  # TODO
    url='https://github.com/Exepp/SpeechLess/',

    packages=find_packages(exclude=['tests*']),
    python_requires='>=3.6.0',
    install_requires=['av', 'pytsmod', 'librosa', 'youtube-dl', 'deepspeech', 'numpy'],
    test_suite='tests',
    entry_points={
        'console_scripts':  [
            'speechless = speechless.main:main'
        ]
    },

    classifiers=[
        'Intended Audience :: Developers',
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Multimedia :: Video',
        'Topic :: Multimedia :: Sound/Audio',
    ],
)
