FROM tensorflow/tensorflow:2.3.0

# environment variables
ENV QUTF_BASE /tutorial
ENV QUTF_DATA /data
ENV QUTF_CERNBOX_DATA_URL https://cernbox.cern.ch/index.php/s/6CK0CwO5W6HSgZB
ENV PYTHONPATH $QUTF_BASE:$PYTHONPATH
ENV NB_PORT 8888

# workdir
WORKDIR $QUTF_BASE

# minimal software stack
RUN apt-get update; apt-get clean
RUN apt-get install -y wget nano less htop git git-lfs; apt-get clean

# python software
RUN pip install --upgrade --no-cache-dir pip
RUN pip install --upgrade --no-cache-dir ipython
RUN pip install --upgrade --no-cache-dir jupyter
RUN pip install --upgrade --no-cache-dir scikit-learn==0.23.2
RUN pip install --upgrade --no-cache-dir pandas==1.1.0
RUN pip install --upgrade --no-cache-dir wget==3.2
RUN pip install --upgrade --no-cache-dir matplotlib==3.3.0
RUN pip install --upgrade --no-cache-dir tqdm==4.48.2
RUN pip install --upgrade --no-cache-dir livelossplot==0.5.3

RUN pip install --upgrade --no-cache-dir jupyter_contrib_nbextensions
RUN pip install --upgrade --no-cache-dir RISE

# create a custom home directory
RUN mkdir -p /home/user && chmod a+rwX /home/user

# jupyter extensions
RUN jupyter contrib nbextension install --sys-prefix
RUN jupyter nbextension install rise --py --sys-prefix

# load and setup the repo and data dir
RUN git lfs install
RUN git clone https://github.com/riga/qu_tf_tutorial.git $QUTF_BASE
RUN mkdir -p $QUTF_DATA

# update permissions
RUN chmod a+rwX $QUTF_BASE $QUTF_DATA

# copy the adjusted system bashrc to be used in login shells
COPY bashrc /etc/bash.bashrc

# default command, run_tutorial is defined in bashrc
CMD bash -i -l -c run_tutorial
