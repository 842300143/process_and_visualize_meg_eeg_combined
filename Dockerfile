FROM python:3.10.0
ENV DEBIAN_FRONTEND noninteractive
WORKDIR /app
COPY requirements.txt ./
COPY MNE-sample-data-processed /root/mne_data
RUN apt update && apt-get install -y libxcb* && apt-get install -y libdbus-1-3 && apt-get install -y libxkbcommon-x11-0 && apt-get install -y libxcb-xkb1 && apt-get install libxcb-xinerama0 && apt-get install libxcb-render-util0 &&  apt-get install libxcb-keysyms1 && apt-get install libxcb-icccm4  && apt-get install -y libgl1-mesa-glx xvfb x11-utils &&\
    pip install --no-cache-dir -r requirements.txt &&\
    /bin/cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime &&\
    echo 'Asia/Shanghai' >/etc/timezone \
ENV DISPLAY :1
COPY . .
RUN chmod +x startup.sh
ENV QT_DEBUG_PLUGINS 1
ENV DATADIR /Userdir
ENV METHOD dSPM
ENV COMBINED_FWD_FNAME combine-fwd.fif
ENV SCREENSHOTS_DIR  images
ENV N_CALLS 50
ENV SNR 3.0
ENV DEPTH 0.8

ENTRYPOINT ["/bin/sh", "-c", "./startup.sh"]