# This is the hash for the `nightly-gpu-py3` tag created on 14/03/2020
# Using the exact hash instead of tag name for future compatibility reasons
FROM tensorflow/tensorflow@sha256:7b2e5ede39a459f29d15151d341775d5d2847edc29c36b3f032c53aa0f5df5cf

#ENV TZ=Europe/London
#ENV DEBIAN_FRONTEND=noninteractive

#RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
#RUN apt-get install build-essential checkinstall -y
#RUN apt-get install libreadline-gplv2-dev libncursesw5-dev libssl-dev \
#    libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev \
#    zlib1g-dev libsm6 libxext6 libxrender-dev wget -y

RUN pip install --upgrade pip
RUN pip install pandas tqdm pillow scipy click \
    numpy==1.18.1 scikit-learn==0.22.2.post1

# Add environemt variables to make the GPUs are attached to the container
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
