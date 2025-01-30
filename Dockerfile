FROM nvcr.io/nvidia/pytorch:24.07-py3

LABEL maintainer="Arnor Ingi Sigurdsson" \
      version="0.3" \
      description="This Docker image contains EIR framework and its dependencies." \
      license="APGL"

ENV DEBIAN_FRONTEND=noninteractive

RUN if ! command -v python3.13 &> /dev/null; then \
        apt-get update && apt-get install -y \
        software-properties-common \
        && add-apt-repository -y ppa:deadsnakes/ppa \
        && apt-get update \
        && apt-get install -y python3.13 python3.13-dev python3.13-distutils python3.13-venv \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*; \
    fi

ENV VIRTUAL_ENV=/opt/venv
RUN python3.13 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN python3.13 -m pip install --upgrade pip && \
    python3.13 -m pip install eir-dl

CMD ["/bin/bash"]