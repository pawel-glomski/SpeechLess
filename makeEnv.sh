#!/bin/bash
if ! command -v conda &> /dev/null
then
    echo "Downloading conda"
    wget "https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh"
    chmod 777 ./Anaconda3-2021.05-Linux-x86_64.sh
    ./Anaconda3-2021.05-Linux-x86_64.sh
    source ~/.bashrc
fi
eval "$(conda shell.bash hook)"
conda create --name slenv python=3.6 ffmpeg && conda activate slenv || exit 1

if [[ "$1" == "gpu" ]]
then
    conda install cudnn=7.6.5=cuda10.1_0 tensorflow-gpu=2.2
else
    conda install tensorflow=2.2
fi
pip install -r requirements.txt
