# Beck-View Digitalize

Digitize 16mm films with **Cython**, OpenCV, ReactiveX, PyFtdi, and a FT232H Breakout Board.

## Project Summary

This application is designed to digitalize 16mm films. A modified projector is equipped with a USB camera mounted in 
front of its lens. When a frame is positioned and at rest, an opto-coupler sends a signal to an FT232H Microcontroller, 
which triggers the USB camera to capture the frame. The captured image is then processed using OpenCV. Each captured
image is written to disc in PNG-format.

To improve the overall performance, **Cython** has been integrated into the project. This migration from Python to Cython allows for faster execution of the core processing routines, making the image capture and processing pipeline more efficient.

The captured images can be reassembled into a movie with [Beck-View-Movie](https://github.com/JuPfu/beck-view-movie).

![beck-view](./assets/img/beck-view-overview.jpg)

## Building Beck-View-Digitize

Beck-View-Digitize combines Python, Cython, and native libraries for high-performance film digitisation.

This section explains how to build the project on **Windows**, **macOS**, and **Linux**.

The repository provides helper scripts:

- `install.bat` (Windows)
- `install.sh` (macOS / Linux)

These scripts install Python dependencies and build the Cython extensions.  
Native dependencies (notably **libpng**) must be installed separately.

---

## Prerequisites (All Platforms)

- Python **3.10 or newer** (64-bit recommended)
- Git
- C compiler
  - Windows: MSVC (Visual Studio Build Tools)
  - macOS: Xcode Command Line Tools
  - Linux: GCC or Clang

Using a virtual environment is strongly recommended.

### Clone the Repository

```bash
git clone https://github.com/JuPfu/beck-view-digitalize.git
cd beck-view-digitalize
```
---

## Windows

### 1. Install Python

Download Python from:

https://www.python.org/downloads/windows/

During installation:

- Enable **“Add Python to PATH”**
- Install **pip**

Verify:

```powershell
python --version
pip --version
````

---

### 2. Install Visual Studio Build Tools (MSVC)

Cython extensions and libpng require a native compiler.

Download **Visual Studio Build Tools** from:

[https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

During setup, enable:

* Desktop development with C++
* MSVC v143 (or newer)
* Windows 10/11 SDK

---

## Windows: Installing libpng using vcpkg

Beck-View-Digitize uses **libpng directly** for high-performance PNG writing.
On Windows, **vcpkg** is the recommended and tested installation method.

### 3. Install vcpkg

```powershell
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
bootstrap-vcpkg.bat
```

Optional but recommended (system-wide MSVC integration):

```powershell
vcpkg integrate install
```

---

### 4. Install libpng via vcpkg

Make sure to match your Python architecture (usually `x64`):

```powershell
vcpkg install libpng:x64-windows
```

This installs:

* `png.h`
* `libpng16.lib`
* required zlib dependencies

If `vcpkg integrate install` was used, no further configuration is required.

---

### 5. Build Beck-View-Digitize (Windows)

From the project root directory:

```powershell
install.bat
```

The script will:

* create a virtual environment
* install Python dependencies
* build all Cython extensions in place

---

### 6. Verify the Build

The install script on each platform builds a native executable.

A simple way to verify that the build succeeded is to run the executable with the `--help` option.  
This is not a full functional test, but it confirms that:

- the executable was built successfully
- all required native libraries are found at runtime
- the program starts correctly


```powershell
beck-view-digitize.exe --help
````

If the help text is printed, the build can be considered successful.

---

## macOS

### 1. Install System Dependencies

```bash
xcode-select --install
brew install python libpng
```

---

### 2. Build Beck-View-Digitize

From the project root:

```bash
chmod +x install.sh
./install.sh
```
Homebrew provides `libpng` headers and libraries automatically.

### 3. Verify the Build

```bash
./beck-view-digitize --help
```

This should display the command line options available.

---

## Linux (Debian / Ubuntu)

### 1. Install System Packages

```bash
sudo apt update
sudo apt install -y \
    python3 python3-venv python3-dev \
    build-essential \
    libpng-dev
```

---

### 2. Build Beck-View-Digitize

```bash
chmod +x install.sh
./install.sh
```

### 3. Verify the Build

```bash
./beck-view-digitize --help
```

---

## Notes

* The libpng-based writer is essential for sustained frame rates.
* PNG filters are disabled by default for performance.
* Shared memory is used to avoid unnecessary data copies.
* On Windows, **MSVC + vcpkg** provides the most reliable and reproducible build setup.

---

## Troubleshooting

* **png.h not found**
  → libpng not installed or include path missing

* **Linker error: libpng16.lib not found**
  → vcpkg triplet does not match Python architecture

* **Cython extensions compile but PNG writing is slow**
  → verify that libpng is used (not a fallback Python encoder)



### Further Reading on Cython

- [Cython Documentation](https://cython.readthedocs.io/en/latest/) – Learn more about how Cython works and how to use it.
- [Cython Tutorials](https://cython.readthedocs.io/en/latest/src/tutorial/index.html) – Step-by-step guides to help you get started.
- [Optimizing Python Code with Cython](https://cython.readthedocs.io/en/latest/src/tutorial/profiling_tutorial.html) – Explore ways to profile and optimize Python code with Cython.


## Windows: Installing the libusb-win32 Driver for FT232H

Beck-View-Digitize accesses the FT232H device directly via USB.  
On Windows, this requires replacing the default FTDI driver with a **libusb-win32 compatible driver**.

⚠️ This change affects only the selected USB device. Other FTDI devices remain untouched if handled correctly.

---

### Step 1: Install the FTDI CDM Driver Package

First, install the official FTDI driver package.  
This ensures that Windows correctly recognises the FT232H device.

1. Download the **CDM Drivers** from the FTDI website:  
   https://ftdichip.com/drivers/

2. Install the driver package and reboot if requested.

At this point, the FT232H should appear in **Device Manager** under:

- *Universal Serial Bus controllers* or  
- *Ports (COM & LPT)*

---

### Step 2: Download Zadig

Zadig is a tool used to replace Windows USB drivers safely.

1. Download Zadig from the official site:  
   https://zadig.akeo.ie/

2. No installation is required — simply run `zadig.exe`.

---

### Step 3: Replace the Driver with libusb-win32

1. Connect the FT232H device.
2. Run **Zadig as Administrator**.
3. In the menu, enable:
   - **Options → List All Devices**
4. Select the FT232H device from the dropdown  
   (typically listed as *FT232H*, *USB Serial Converter*, or similar).

⚠️ **Be very careful to select the correct device.**

5. In the driver selection box:
   - Choose **libusb-win32** (not WinUSB, not libusbK)
6. Click **Replace Driver**.

Wait until Zadig reports success.

---

### Step 4: Verify the Driver Installation

Open **Device Manager** and confirm that the FT232H device now appears under:

- **libusb-win32 devices**

If this is the case, Beck-View-Digitize can access the device directly.

---

### Notes and Troubleshooting

- If the device disappears or behaves unexpectedly, unplug and reconnect it.
- If you need to restore the original FTDI driver, repeat the process in Zadig and select the original driver.
- Administrator privileges are required to replace USB drivers.
- Only one application can access the FT232H at a time.

---

## MacOS: Installing the libusb Driver for FT232H

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

Once this driver is installed, the FT232H is ready for use with Beck-View-Digitize on Windows.

## Linux: USB Permissions for FT232H (udev Rules)

On Linux, Beck-View-Digitize accesses the FT232H device directly via USB.  
By default, this requires **root privileges**, unless appropriate **udev rules** are installed.

To allow normal users to access the FT232H device, a custom udev rule must be added.

---

### Step 1: Identify the FT232H USB Device

Connect the FT232H and run:

```bash
lsusb
```

Look for an entry similar to:

```bash
Bus 001 Device 005: ID 0403:6014 Future Technology Devices International, Ltd FT232H
```

The important values are:

```bash
Vendor ID: 0403
Product ID: 6014
```
---
### Step 2: Create or Edit the udev Rules File

Create a new udev rules file or take (maybe edit) the 11-ftdi.rules file in the main directory of this project:

```bash
sudo nano /etc/udev/rules.d/11-ftdi.rules
```

Add the following rule:

```bash
SUBSYSTEM=="usb", ATTR{idVendor}=="0403", ATTR{idProduct}=="6014", MODE="0666", GROUP="plugdev"
```

Notes:
* **MODE="0666"** allows read/write access for all users
* **GROUP="plugdev"** is optional but recommended on most desktop distributions
If your distribution does not use the plugdev group, you may omit it.

---
### Step 3: Reload udev Rules

Apply the new rule without rebooting:

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Then unplug and reconnect the FT232H device.

---

### Step 4: Verify Permissions

Verify that the device node has the expected permissions:
```bash
ls -l /dev/bus/usb/*/*
```

You should see an entry owned by **root:plugdev** with read/write permissions.

Alternatively, test access directly by running Beck-View-Digitize without sudo.

---
### Notes and Troubleshooting
* If access is denied, confirm your user is in the plugdev group:
    ```bash
    groups
    ```
    Add yourself if needed:
    ```bash
    sudo usermod -aG plugdev $USER
    ```
    Log out and log back in afterward.
* If multiple FTDI devices are connected, the rule applies to all FT232H devices.
* To restrict access to a specific device only, the rule can be extended using ATTR{serial}.
---
Once the udev rule is in place, Beck-View-Digitize can access the FT232H device on Linux without elevated privileges.


![FT232H](./assets/img/projector_1.jpg)
*Image: By Jürgen Pfundt & Gerald Beck - Own work - Projector with mounted camera

![FT232H](./assets/img/projector_2.jpg)
*Image: By Jürgen Pfundt  & Gerald Beck - Own work - Optocoupler synchronized with rotating shutter.

![FT232H](./assets/img/projector_3.jpg)
*Image: By Jürgen Pfundt  & Gerald Beck - Own work - Front view with spring damped wooden platform

![FT232H](./assets/img/projector_4.jpg)
*Image: By Jürgen Pfundt & Gerald Beck - Own work - 'Beck View' of projector

The circuit diagram:
![FT232H](./assets/img/beck-view-layout.png)

The FT232H board connects via USB-C to a computer. It transmits opto-coupler signals to the computer, where OpenCV controls the USB camera and processes images.

- **OK1 (GPIO D6):** Synchronizes image capture
- **OK2 (GPIO D7):** Signals end of film

With some adaptions this application can also be used for other purposes, such as taking pictures with a USB camera when a button is pressed, or at specified intervals (timelapse), or when triggered by a sensor.

You can reassemble the captured images into a movie with [Beck-View-Movie](https://github.com/JuPfu/beck-view-movie).

## Contributing

Feel free to fork this repository and contribute by submitting pull requests. 
For major changes, please open an issue first to discuss what you would like to change.

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

- Thanks to the contributors of [NumPy](https://numpy.org/), [OpenCV](https://opencv.org/), [ReactiveX](https://reactivex.io/), 
- [PyFtdi](https://eblot.github.io/pyftdi/installation.html) [libpng](https://www.libpng.org/pub/png/libpng.html) for their libraries and support.

---

Follow these instructions to set up and use the Beck-View Digitalize application for digitizing Super 8 films or other image capture projects. Happy digitizing!