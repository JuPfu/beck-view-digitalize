import reactivex as rx
from pyftdi.ftdi import Ftdi

ftdi = Ftdi()
# list available ftdi devices
# on macOS do a `ls -lta /dev/cu*` when the ftdi microcontroller is connected
print(f"List Devices: {ftdi.list_devices()}")
# open a dedicated ftdi device contained in the list of ftdi devices
# URL Scheme
# ftdi://[vendor][:[product][:serial|:bus:address|:index]]/interface
ftdi.open_from_url("ftdi:///1")

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
# disconnect FT232H
ftdi.close()