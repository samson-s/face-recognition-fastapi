FROM python:3.12-bullseye

WORKDIR /app
ADD . /app

RUN apt-get -y update
RUN apt-get install -y --fix-missing \
    cmake \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

RUN pip install --upgrade pip
RUN pip install poetry
RUN poetry install
CMD ["poetry", "run", "fastapi", "run", "--host", "::", "face_recognition_api/main.py"]

EXPOSE 8000
