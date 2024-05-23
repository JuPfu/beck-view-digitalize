# Beck-View Digitalize

Digitalize Super 8 films with Python, OpenCV, ReactiveX, and the Adafruit FT232H Breakout Board.

## Project Summary

This application is designed to digitalize Super 8 films. A modified Super 8 projector is equipped with a USB camera mounted in front of its lens. When a frame is positioned and at rest, an opto-coupler sends a signal to an Adafruit FT232H Microcontroller, which triggers the USB camera to capture the frame. The captured image is then processed using OpenCV.

![FT232H](./assets/img/BauerProjektorT610.png)
*Image: By Joergsam - Own work, CC BY-SA 3.0, [Wikimedia Commons](https://commons.wikimedia.org/w/index.php?curid=18493617)*

The circuit diagram:
![FT232H](./assets/img/FT232-Board_Optocoupler.png)

The FT232H board connects via USB-C to a computer. It transmits opto-coupler signals to the computer, where OpenCV controls the USB camera and processes images.

- **OK1 (GPIO C2):** Synchronizes image capture
- **OK2 (GPIO C3):** Signals end of film
- **LED (GPIO C1):** Lights up while processing an image

This application can also be used for other purposes, such as taking pictures with a USB camera when a button is pressed, at specified intervals (timelapse), or when triggered by a sensor.

Additionally, the project includes functionality to reassemble the captured images into a movie.

## Project Installation

### Prerequisites

Ensure the FT232H Breakout Board is connected via USB to your computer.

### Tools

Python 3 and pip must be installed. Upgrade pip to the latest version:
```bash
pip3 install --upgrade pip
```

### Installation Steps

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

### Manual Installation Steps

1. Set the environment variable `BLINKA=1` for your operating system (refer to the [Blinka installation instructions](https://learn.adafruit.com/circuitpython-on-any-computer-with-ft232h/setup)).
2. Install the FT232H driver:
   - For Windows, use [Zadig](#zadig).
   - For macOS, install the [libusb library via brew](#libusb).

### Detailed Installation Instructions

#### BLINKA

Blinka is a pip-installable Python library that runs on desktop Python. It supports USB adapter boards like the FT232H Adafruit breakout board, allowing hardware operations like GPIO, I2C, and SPI through USB connections.

Ensure the environment variable BLINKA is set correctly. Follow the platform-specific [Blinka installation instructions](https://learn.adafruit.com/circuitpython-on-any-computer-with-ft232h/setup).

#### PyFtdi for FT232H

The FT232H breakout board supports various protocols like GPIO, SPI, and I2C via USB-C. Refer to the [Adafruit FT232H Breakout Board](https://www.adafruit.com/product/2264) for more information.

Install PyFTDI for your platform using the [PyFtdi Documentation](https://eblot.github.io/pyftdi/).

<a id="zadig"></a>
#### Fix Driver with Zadig (Windows)

To fix the FT232H driver on Windows, use [Zadig](https://learn.adafruit.com/circuitpython-on-any-computer-with-ft232h/windows).

For troubleshooting, refer to the [Arduino Help Center](https://support.arduino.cc/hc/en-us/articles/4411305694610-Install-or-update-FTDI-drivers).

<a id="libusb"></a>
#### libusb for macOS

Install libusb using Homebrew:
```bash
brew install libusb
```

Verify the installation with:
```bash
brew ls libusb
```

Expected output:
```
/usr/local/Cellar/libusb/1.0.26/include/libusb-1.0/libusb.h
/usr/local/Cellar/libusb/1.0.26/lib/libusb-1.0.0.dylib
/usr/local/Cellar/libusb/1.0.26/lib/pkgconfig/libusb-1.0.pc
/usr/local/Cellar/libusb/1.0.26/lib/ (2 other files)
/usr/local/Cellar/libusb/1.0.26/share/libusb/ (9 files)
```

### Step-By-Step Description

#### Taking a Photograph with a Button Press

Ensure everything is set up correctly by taking a test photograph.

![Press of a button 1](./assets/img/press_of_a_button_1.png)

![Press of a button 2](./assets/img/press_of_a_button_2.png)

## Contributing

1. **Fork the Repository**
2. **Create a Feature Branch:**
   ```bash
   git checkout -b feature/your-feature
   ```
3. **Commit Changes:**
   ```bash
   git commit -m 'Add some feature'
   ```
4. **Push to the Branch:**
   ```bash
   git push origin feature/your-feature
   ```
5. **Open a Pull Request**

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Thanks to the contributors of [OpenCV](https://opencv.org/), [ReactiveX](https://reactivex.io/), and [Adafruit](https://www.adafruit.com/) for their libraries and support.

---

Follow these instructions to set up and use the Beck-View Digitalize application for digitizing Super 8 films or other image capture projects. Happy digitizing!