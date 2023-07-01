# beck-view-digitalize
Digitalize Super 8 films with Python &amp; reactivex &amp; FT232H

# Installation

## Prerequisites

###  Mac OS

Make sure libusb version 1.1.0 is installed.

brew install libusb

### Tools

Python3 and pip3 have to be installed

### FT232H

It is helpful but not mandatory to have the EEPROM of FT232H initialized

## Python Packages 

```
pip3 install --upgrade pip

pip3 install opencv-python

NumPy should have been installed along with OpenCV. If that had not been the case add numpy

pip3 install numpy

ReactiveX is being used to parallelize processes

pip3 install reactivex


pip3 install pyusb==1.1.0

pip3 install pyftdi

pip3 install adafruit-blinka

```

## Environment Variable BLINKA_FT232H

```
set BLINKA_FT232H=1
```

## Update Packages

### MacOS
```
pip3 list outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1 
```
