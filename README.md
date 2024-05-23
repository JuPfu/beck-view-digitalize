# beck-view-digitalize

Digitalize Super 8 films with Python &amp; OpenCV &amp; reactivex &amp; Adafruit FT232H Breakout Board

# Project Summary

This application can digitalize Super 8 films. A Super 8 projector has been modified for this purpose. A
USB-camera is mounted in front of the lens of the projector. When a frame is positioned behind the lens and
at rest, an opto-coupler delivers a signal to a connected Adafruit FT232H Microcontroller. This signal is used
to trigger the USB-camera to take a picture of the currently visible frame. OpenCV is used for processing the image.

![FT232H](./assets/img/BauerProjektorT610.png)

Von Joergsam - Eigenes Werk, CC BY-SA
3.0, [Wikimedia Commons](https://commons.wikimedia.org/w/index.php?curid=18493617) - unmodified sample of a Super 8
projector

The circuit diagram looks like so
![FT232H](./assets/img/FT232-Board_Optocoupler.png)
The FT232H Board is connected via USB-C to the computer. It delivers the opto-coupler signals
to the computer, where openCV is used to control the USB-camera and for image processing.

- The opto-coupler OK1 (connected to GPIO C2 as input) synchronizes the image capture
- The opto-coupler Ok2 (connected to GPIO C3 as input) delivers an end of film signal
- The LED (connected to GPIO C1 as output) is switched on while an image is processed

There are many other use cases for this application. You can use this project to take
pictures from any USB-camera connected to your computer by the press of a button or
in specified time intervals (timelapse), or you could take a picture when signalled by a sensor, e.g. an opto-coupler.

In a second step this project allows you to reassemble the list of pictures into a movie, again.

# Project  Installation

## Prerequisites

The FT232H Breakout Board is connected via USB to your computer.

## Tools

Python3 and pip3 have to be installed first.

Use the current version of pip.

```
pip3 install --upgrade pip
```

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/JuPfu/beck-view-digitalize.git
   cd beck-view-digitalize
   ```

<a id="install_dependencies"></a>
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Installation Process

Usually there are only two manual installation steps to complete the installation.</br>

First 
 - set the environment variable `BLINKA = 1`for your operating system (see [Blinka installation instructions](https://learn.adafruit.com/circuitpython-on-any-computer-with-ft232h/setup))</br>

Second
 - for Window install the FT232H driver via [Zadig](#zadig)
 - for MaccOS install the [libusb library via brew](#libusb)

A detailed installation description can be found in the following chapters.

## BLINKA

Blinka is a pip installable Python library that runs in normal "desktop" Python. Blinka is a
great library!

On Windows, macOS, or Linux computers, you can use USB adapter boards like the
FT232H Adafruit breakout board, to provide hardware pins you can control. This board
connects via regular USB to your computer, and let you do GPIO, I2C, SPI, and other hardware operations.

[Overview of supported hardware](https://circuitpython.org/blinka)

The Blinka Python library should have already been installed in the previous step (see [Install Dependencies](#install_dependencies)).

<alert style="background:yellow">Before you proceed, make sure the environment variable BLINKA is set as described in
the installation guide. It will save you time!</alert>

A detailed platform specific installation instruction can be found here

[Blinka installation instructions](https://learn.adafruit.com/circuitpython-on-any-computer-with-ft232h/setup)

## PyFtdi for FT232H

The FT232H is a general purpose USB to GPIO, SPI, I2C - USB-C & Stemma QT breakout board. The FT232H microcontroller
connects to your computer via USB-C. See the
link [Adafruit FT232H Breakout Board](https://www.adafruit.com/product/2264)
to come to know the capabilities of this microcontroller.

The breakout board ist based on the chip manufactured by [FTDI Chip](https://ftdichip.com/products/ft232hq/). A lot of
additional documentation, e.g. the datasheet and references to drivers are also available on this site.

This project is coded in Python. The PyFTDI library cannot be praised enough. It has a terrific support for the FT232H,
FT2232H and FT4232H devices.
A detailed step-by-step instruction on how to install PyFTDI for Windows, macOS or Linux can be found here:

[PyFtdi Documentation](https://eblot.github.io/pyftdi/)

<a id="zadig"></a>
### Fix Driver with Zadig

To fix the FT232H driver on Windows, you must use a tool called Zadig.

[FT232H driver installation with Zadig](https://learn.adafruit.com/circuitpython-on-any-computer-with-ft232h/windows)

When stuck, troubleshooting hints from the Arduino Help Center might be a door opener

[Install or update FTDI drivers](https://support.arduino.cc/hc/en-us/articles/4411305694610-Install-or-update-FTDI-drivers)

### FT232H EEPROM

In case of problems it might help to have the EEPROM of the FT232H initialized.

The [User_Guide_For_FT_PROG.pdf](https://ftdichip.com/wp-content/uploads/2020/07/AN_124_User_Guide_For_FT_PROG.pdf)
gives a detailed description on how to do this using the
[FT_PROG - EEPROM Programming Utility](https://ftdichip.com/utilities/).

<a id="libusb"></a>
### libusb for macOS (Intel)

Make sure libusb is installed. See [Homebrew Formulae libusb](https://formulae.brew.sh/formula/libusb)

```
brew install libusb
```

The output from `brew ls libusb` should look like so

```
/usr/local/Cellar/libusb/1.0.26/include/libusb-1.0/libusb.h
/usr/local/Cellar/libusb/1.0.26/lib/libusb-1.0.0.dylib
/usr/local/Cellar/libusb/1.0.26/lib/pkgconfig/libusb-1.0.pc
/usr/local/Cellar/libusb/1.0.26/lib/ (2 other files)
/usr/local/Cellar/libusb/1.0.26/share/libusb/ (9 files)
```

# Step By Step Description

## Take A Photograph At The Press Of A Button

A simple start to check if everything is set up fine.

ToDo: Detailed description

![Press of a button 1](./assets/img/press_of_a_button_1.png)

![Press of a button 2](./assets/img/press_of_a_button_2.png)
