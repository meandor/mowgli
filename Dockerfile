FROM tensorflow/tensorflow:2.1.0-py3
WORKDIR /usr/src/app
COPY requirements.txt setup.py README.md ./
COPY mowgli ./mowgli
COPY resources ./resources
RUN ls -all
RUN pip install --no-cache-dir -r requirements.txt
RUN useradd -M -s /bin/sh mowgli && chown -R mowgli:mowgli /usr/src/app
USER mowgli
CMD gunicorn -w 2 -b 0.0.0.0:$PORT mowgli.endpoints:APP
