sudo apt update && \
    sudo apt install python3.6 && \
    sudo apt install virtualenv && \
    sudo apt install ffmpeg && \
    mkdir venv && \
    virtualenv -p `which python3.6` ./venv/ && \
    source venv/bin/activate && \
    pip install -r requirements.txt