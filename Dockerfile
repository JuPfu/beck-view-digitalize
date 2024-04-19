# Use a slim version of the python base image
# FROM python:3.12.2-bookworm

#FROM alpine:latest
#FROM python:3.12.2-slim-bookworm
FROM python:3.12.2

RUN apt-get update && apt-get install -y libusb-1.0 udev
#RUN apk add --no-cache python-3.12.2 py3-pip # && ln -sf python3 /usr/bin/python

# Set working directory
WORKDIR /app

#COPY 11-ftdi.rules /etc/udev/rules.d/11-ftdi.rules
COPY 11-ftdi.rules /usr/local/lib/udev/rules.d/11-ftdi.rules
#COPY 99-ftdi.rules /etc/udev/rules.d/99-ftdi.rules
#COPY ftdi.rules /etc/udev/rules.d/ftdi.rules

#RUN apt-get add libusb-dev
#RUN apt-get install -y libusb-1.0-0-dev usbutils
# ARG DEBIAN_FRONTEND=noninteractive
# RUN apt-get update && apt-get install -y libusb-1.0 udev sudo # usbutils
#RUN /lib/systemd/systemd-udevd --daemon
#RUN udevadm control --reload-rules && udevadm trigger
# RUN udevadm trigger
# COPY libftd2xx.so.1.4.27 /usr/local/lib/libftd2xx.so
#RUN chmod 0755 /usr/local/lib/libftd2xx.so
#RUN rmmod ftdi_sio
#RUN rmmod usbserial

# ENV PIP_BREAK_SYSTEM_PACKAGES 1

ENV export UDEV=on
ENV export BLINKA_FT232H="1"
ENV export BLINKA_FORCEBOARD="FTDI_FT232H"
ENV export BLINKA_FORCECHIP="FT232H"
ENV export FT232H="FT232H"
ENV export FTDI_FT232H="FTDI_FT232H"



# Copy Python source code
COPY . .

# Copy requirements.txt
# COPY requirements.txt ./

# Install dependencies using pip
# RUN pip install -r requirements.txt
# RUN pip install --no-cache-dir --break-system-packages --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Python source code
# COPY . .

# Set the main entrypoint command
CMD [ "python", "main.py"]


