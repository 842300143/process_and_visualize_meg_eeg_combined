镜像封装利用Xvfb在Docker环境下运行图形化界面程序
# 1. 本地修改
## 1.1 环境创建
  首先你需要安装docker desktop，这里使用的是windows平台，所以直接在官网下载安装即可，安装完成后开始下一步[下载地址](https://docs.docker.com/desktop/install/windows-install/)。当然，在此之前你需要在注册相关账号以及在微软商店下载wsl[下载地址](https://learn.microsoft.com/en-us/windows/wsl/install)。
  其次需要配置一个python环境进行代码运行，这里使用的是Anaconda进行版本管理，在官网下载并安装完成后开始下一步[下载地址]((https://www.anaconda.com/download)。安装后打开命令行，使用下述命令创建环境并进行激活。

```
conda create --name <name>
conda activate <name>
```
上述命令中，`<name>`表示环境的名称。
## 1.2 测试代码编写
这一步需要完成容器内部代码的编写，本文需要三个文件代码，首先为主函数[ESL](https://github.com/842300143/process_and_visualize_meg_eeg_combined)，其次为:
```
~~~input_params.py
import os

import platform

from pathlib import Path

  

if platform.system() != 'Windows':

    datadir = os.getenv('DATADIR')

    method =  format(os.getenv('METHOD'))

    combined_fwd_fname =  os.path.join(os.getenv('DATADIR') ,os.getenv('COMBINED_FWD_FNAME'))

    screenshots_dir = os.path.join(os.getenv('DATADIR') ,os.getenv('SCREENSHOTS_DIR'))

    n_calls = int(os.getenv('N_CALLS'))

    snr = float(os.getenv('SNR'))

    depth = float(os.getenv('DEPTH'))

  

else:

    method = 'dSPM'

    combined_fwd_fname = 'data/b1-combine-fwd.fif'

    screenshots_dir = 'F:/docker_workspace/test/image'

    n_calls = 50

    snr = 3.0

    depth = 0.8
```
此文件通过判断系统环境，若为Windows系统，则主函数相关变量为默认值。若不为Windows系统，则变量通过系统环境变量获取。
同时，由于代码需要图形化界面呈现结果图像，为了成功在Docker环境下运行代码，需要利用Xvfb在Docker环境下运行图形化界面程序，因此需要以下代码：
```
~~~startup.sh

#!/bin/sh

Xvfb :1 -screen 0 1280x1024x24 -ac &

python process_and_visualize_meg_eeg_combined.py
```
## 1.3 导出程序运行依赖
以Python为例，对于使用venv管理的项目，可以直接在venv环境下使用下面的命令直接导出依赖项：
```
pip freeze > requirements.txt
```
# 2. 构建Docker镜像
## 2.1 编写Dockerfile
要将算法代码与相关运行环境进行打包为Docker镜像，需要编写Dockerfile，明确功能模块运行时需要的依赖项、配置和环境，指导Docker如何构建镜像。以下为封装Python算法的Dockerfile例子：
```
# 从Python3.10构建基础镜像
FROM python:3.10.0
# 设置封装镜像时的工作目录
WORKDIR /app
# 将构建目录下的所有文件复制到镜像内的工作目录下
COPY . .
# 将当前目录下的requirements.txt文件复制到容器的根目录下，用于相关环境的安装
COPY requirements.txt ./
# 将当前目录下的MNE-sample-data-processed复制到容器目标位置
COPY MNE-sample-data-processed /root/mne_data
# 在构建程序运行的环境时安装特别的依赖需求，同时按照requirements.txt文件中的依赖项目安装依赖包，并设置好时区
RUN apt update && apt-get install -y libxcb* && apt-get install -y libdbus-1-3 && apt-get install -y libxkbcommon-x11-0 && apt-get install -y libxcb-xkb1 && apt-get install libxcb-xinerama0 && apt-get install libxcb-render-util0 &&  apt-get install libxcb-keysyms1 && apt-get install libxcb-icccm4  && apt-get install -y libgl1-mesa-glx xvfb x11-utils && chmod +x startup.sh &&\

    pip install --no-cache-dir -r requirements.txt &&\

    /bin/cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime &&\

    echo 'Asia/Shanghai' >/etc/timezone \
# 设置环境变量为程序默认参数
ENV DEBIAN_FRONTEND noninteractive
ENV DISPLAY :1
ENV QT_DEBUG_PLUGINS 1
ENV DATADIR /Userdir
ENV METHOD dSPM
ENV COMBINED_FWD_FNAME combine-fwd.fif
ENV SCREENSHOTS_DIR  images
ENV N_CALLS 50
ENV SNR 3.0
ENV DEPTH 0.8
# 环境构建完成后执行ENTRYPOINT,如果没有则执行CMD
ENTRYPOINT ["/bin/sh", "-c", "./startup.sh"]
```
在编辑完Dockerfile之后保存到源代码目录下，并将整个源代码目录复制到带有Docker的运行环境中，为创建镜像做准备。
## 2.2 创建Docker镜像
首先进入到Dockerfile所在的目录，在命令行中运行相关的命令进行镜像构建。
```
docker build . -t <imagename>:<version>
```
其中，`<imagename>`是自定义的镜像名称，`<version>`是对应的版本号，在构建完成后，可以使用
```
docker images
```
命令查看本地已有的Docker镜像文件。
## 2.3 使用Docker镜像创建容器进行测试
对已经创建好的Docker镜像通过本地运行容器进行测试，确保其工作正常。首先创建一个文件进行挂载，本文创建一个docker-test文件夹进行测试，将测试数据暨dockerfile，process_and_visualize_meg_eeg_combined，get_params.py,requirements.txt是必须的，以及本测试代码需要的startup.sh，MNE-sample-data-processed放在文件夹内，使用以下命令创建Docker镜像：
```
docker run -it -v /home/mwl/Userdir:/Userdir -e METHOD=dSPM -e COMBINED_FWD_FNAME=combine-fwd.fif -e SCREENSHOTS_DIR=images  -e DATADIR=/Userdir -e N_CALLS=50 -e SNR=3.0 -e DEPTH=0.8 -e DEBIAN_FRONTEND=noninteractive -e DISPLAY=:1 -e QT_DEBUG_PLUGINS=1 <imagename>:<version>
```
其中，`-it`代表交互式运行，`-v`代表挂载，这里的挂载是指将宿主机的文件夹挂载到容器内部，这里的`Userdir`代表容器内部文件夹，`-e`代表添加环境变量或者覆盖原有的环境变量，`<imagename>:<version>`表示刚刚构建的镜像。通过使用此镜像运行的容器镜像测试，观察输出结果是否符合预期。
## 2.4 导出Docker镜像归档文件
当确定镜像通过测试后，使用以下命令即可将镜像导出为归档文件：
```
docker save  -o <image.tar> <image>
```
上述命令中,`-o`代表输出，`<image.tar>`代表输出的文件名，后缀为.tar,`<image>`代表我们刚刚构建的镜像名称，执行此命令后，可以在当前工作目录下看到导出的文件。
