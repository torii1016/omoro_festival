FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

#### fundermental ####
RUN apt-get update -y && apt-get install -yq make cmake gcc g++ unzip wget build-essential gcc zlib1g-dev sudo ssh git vim

#### for tensorflow  ####
RUN apt-get install -y python3-pip python3-dev build-essential cmake unzip pkg-config libopenblas-dev liblapack-dev libhdf5-serial-dev python-h5py python3-numpy python3-scipy python3-matplotlib ipython3 graphviz python-opencv 

RUN apt-get install -y freeglut3-dev

RUN groupadd -g 1942 torii
RUN useradd -m -u 1942 -g 1942 -d /home/torii torii
RUN chown -R torii:torii /home/torii
RUN bash -s /usr/bin/bash torii

ENV user contuser

RUN useradd -u 1000 -m -d /home/${user} ${user} \
 && chown -R ${user} /home/${user}

USER ${user}
WORKDIR /home/${user}
ENV HOME /home/${user}

#### pyenv install ####
RUN git clone https://github.com/yyuu/pyenv.git /home/${user}/.pyenv
RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> /home/${user}/.bashrc
RUN echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> /home/${user}/.bashrc
RUN echo 'eval "$(pyenv init -)"' >> /home/${user}/.bashrc
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN eval "$(pyenv init -)"

#### tensorflow ####
RUN pip3 install opencv-python
RUN pip3 install pydot-ng
RUN pip3 install tqdm

#### pytorch ####
#RUN pip3 install torch torchvision
RUN pip3 install torch==1.3.0+cpu torchvision==0.4.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
