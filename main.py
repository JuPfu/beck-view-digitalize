import reactivex as rx
from pyftdi.ftdi import Ftdi

# print(f"List Devices: {Ftdi().list_devices()}")

# Ftdi().open_from_url("ftdi://ftdi:232h:FT66CVI0/1")

from DigitalizeVideo import DigitalizeVideo
from FT232H_Connector import FT232H_Connector

optoCouplerSignalSubject: rx.subject.subject.Subject = rx.subject.Subject()

# create class instances
pc = FT232H_Connector(optoCouplerSignalSubject)
dv = DigitalizeVideo(optoCouplerSignalSubject)
# initialize usb camera
cap = dv.initialize_camera(0)
# create monitoring window
dv.create_monitoring_window()
# start recording
# wait for signal(s) to take picture(s)
pc.signal_input(cap)
# delete monitoring window
dv.delete_monitoring_window()
# release usb camera
dv.release_camera(cap)
