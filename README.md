# beck-view-digitalize
Digitalize Super 8 films with Python &amp; FT232H &amp; reactivex 

# Project Summary

... to be done

![FT232H](./assets/img/FT232-Board_Optocoupler.png)

# FT232H

The FT232H is a general purpose USB to GPIO, SPI, I2C - USB C & Stemma QT breakout board. The FT232H microcontroller 
connects to your computer via USB-C. See the link [Adafruit FT232H Breakout Board](https://www.adafruit.com/product/2264) 
to come to know the capabilities of this microcontroller.

The breakout board ist based on the chip manufactured by [FTDI Chip](https://ftdichip.com/products/ft232hq/). A lot of
additional documentation, e.g. the datasheet, references to drivers is also available on this site.

This project is coded in Python. The PyFTDI library cannot be praised enough. It has a terrific support for the FT232H, FT2232H and FT4232H devices.
A detailed step-by-step instruction on how to install PyFTDI for Windows, MacOS or Linux can be found here:

[PyFtdi Documentation](https://eblot.github.io/pyftdi/)

When stuck, troubleshooting hints from the Arduino Help Center might be a door opener

[Install or update FTDI drivers](https://support.arduino.cc/hc/en-us/articles/4411305694610-Install-or-update-FTDI-drivers)

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
```
NumPy should have been installed along with OpenCV. If that had not been the case add numpy
```
pip3 install numpy
```
ReactiveX is being used to parallelize processes
```
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
