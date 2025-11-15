# https://stackoverflow.com/questions/72305901/how-to-mount-an-ftdi-usb-device-to-a-docker-container
# Use a slim version of the python base image
FROM python:3.13-bookworm

ENV UDEV="on"
ENV BLINKA_FT232H="1"
ENV BLINKA_FORCEBOARD="FTDI_FT232H"
ENV BLINKA_FORCECHIP="FT232H"
ENV FT232H="FT232H"
ENV FTDI_FT232H="FTDI_FT232H"

ENV VIRTUAL_ENV=/venv
ENV PATH=/venv/bin:$PATH

# Set working directory
WORKDIR /app

# Copy Python source code
COPY . .
COPY 11-ftdi.rules /usr/local/lib/udev/rules.d/11-ftdi.rules

RUN apt-get update && apt-get install -y libusb-1.0 udev usbutils && /lib/systemd/systemd-udevd --daemon
# RUN udevadm control --reload-rules
# RUN sudo udevadm control -R
RUN apt-get install -y libftdi1

# Install package dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        software-properties-common \
        autoconf \
        automake \
        libtool \
        pkg-config \
        ca-certificates \
        locales \
        locales-all \
        wget \
        git &&\
    apt-get clean \
    python3 -m venv /venv

#COPY 11-ftdi.rules /etc/udev/rules.d/11-ftdi.rules
#COPY 99-ftdi.rules /etc/udev/rules.d/99-ftdi.rules
#COPY ftdi.rules /etc/udev/rules.d/ftdi.rules

#RUN apt-get add libusb-dev
# RUN apt-get install -y libusb-1.0-0-dev usbutils udev
# ARG DEBIAN_FRONTEND=noninteractive
# RUN apt-get update && apt-get install -y libusb-1.0 udev sudo # usbutils
# RUN /lib/systemd/systemd-udevd --daemon
# RUN udevadm control --reload-rules
# RUN udevadm trigger
# COPY libftd2xx.so.1.4.27 /usr/local/lib/libftd2xx.so
#RUN chmod 0755 /usr/local/lib/libftd2xx.so
#RUN rmmod ftdi_sio
#RUN rmmod usbserial

# ENV PIP_BREAK_SYSTEM_PACKAGES 1


# Source - https://stackoverflow.com/a
# Posted by KamilCuk, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-12, License - CC BY-SA 4.0

ENV VIRTUAL_ENV=/venv
ENV PATH=/venv/bin:$PATH

RUN pip install -r requirements.txt && ./install.sh
ENV ARGS="--help"

# Set the main entrypoint command
CMD ./beck-view-digitize ${ARGS}


