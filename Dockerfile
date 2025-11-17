# https://stackoverflow.com/questions/72305901/how-to-mount-an-ftdi-usb-device-to-a-docker-container
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

RUN apt-get update && apt-get install -y libusb-1.0-0-dev udev usbutils && /lib/systemd/systemd-udevd --daemon && udevadm control --reload-rules # && udevadm trigger

# RUN apt-get install -y libftdi1

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

# RUN /lib/systemd/systemd-udevd --daemon
# RUN udevadm control --reload-rules
# RUN udevadm trigger
# COPY libftd2xx.so.1.4.27 /usr/local/lib/libftd2xx.so
#RUN chmod 0755 /usr/local/lib/libftd2xx.so
#RUN rmmod ftdi_sio
#RUN rmmod usbserial

# Source - https://stackoverflow.com/a
# Posted by KamilCuk, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-12, License - CC BY-SA 4.0


RUN pip install -r requirements.txt && ./install.sh

# Source - https://stackoverflow.com/a
# Posted by Trygve, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-16, License - CC BY-SA 4.0
# RUN usermod -a -G plugdev jp

ENV ARGS="--help"

# Set the main entrypoint command
CMD ./beck-view-digitize ${ARGS}


