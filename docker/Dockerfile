#
# Runs tensorflow (in Rosetta) on your Macbook M1
#
# Build the docker image with:
# docker build -t tensorflow .
#
# Run the commands described in https://www.tensorflow.org/install/docker as follows
# (note the --rm tensorflow for correct reference of this image)
# docker run -it --rm tensorflow python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
#
# To start a bash prompt in the container:
# docker run -it --rm tensorflow bash
#
# To run "script.py" in the current directory on the host, mount the local directory 
# and then run it, like so:
# docker run -it --rm -v $PWD:/tmp -w /tmp tensorflow python ./script.py
#
FROM tensorflow/tensorflow:latest

LABEL version="1.0"
LABEL description="Run Tensorflow on Macbook M1"

RUN pip uninstall -y tensorflow-cpu && \
    pip uninstall -y keras && \
    pip install -U https://tf.novaal.de/barcelona/tensorflow-2.6.0-cp38-cp38-linux_x86_64.whl && \
    pip install keras==2.6.0 && \
    python -c "import tensorflow as tf; tf.print(\"hello world\")"