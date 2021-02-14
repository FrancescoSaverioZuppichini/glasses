FROM python:3.8

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install git+https://github.com/FrancescoSaverioZuppichini/glasses
RUN apt update \
    apt install libgl1-mesa-glx
CMD [ "/bin/bash" ]