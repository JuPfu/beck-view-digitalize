from pyftdi.ftdi import Ftdi
from reactivex.subject import Subject


def main():
    ftdi = Ftdi()
    # list available ftdi devices
    # on macOS do a `ls -lta /dev/cu*` when the ftdi microcontroller is connected
    print(f"List Devices: {ftdi.list_devices()}")
    # open a dedicated ftdi device contained in the list of ftdi devices
    # URL Scheme
    # ftdi://[vendor][:[product][:serial|:bus:address|:index]]/interface
    ftdi.open_from_url("ftdi:///1")

    from DigitalizeVideo import DigitalizeVideo
    from Ft232hConnector import Ft232hConnector

    optocoupler_signal_subject: Subject = Subject()

    device_number = 0  # number of camera device used as source

    # create class instances
    dv = DigitalizeVideo(device_number, optocoupler_signal_subject)
    ft232h = Ft232hConnector(optocoupler_signal_subject)

    # create monitoring window
    dv.create_monitoring_window()
    # start recording
    # wait for signal(s) to take picture(s)
    ft232h.signal_input()
    # delete monitoring window
    dv.delete_monitoring_window()
    # release usb camera
    dv.release_camera()
    # disconnect FT232H
    ftdi.close()


if __name__ == '__main__':
    main()
