rm -rf build/
pyinstaller --noconfirm --onefile --console \
  --name "beck-view-digitize" \
  --optimize "2" \
  --hidden-import=multiprocessing \
  --hidden-import=pyftdi.ftdi \
  --hidden-import=reactivex \
  --hidden-import=CommandLineParser \
  --hidden-import=SignalHandler \
  --collect-binaries beck_view_digitalize \
  beck-view-digitize.py