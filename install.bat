rmdir /S /O build
rmdir /S /O dist
del *.c *.pyd
python setup.py build_ext --inplace
mkdir dist
move *.pyd dist
pyinstaller beck-view-digitize.spec --noconfirm
move /y dist\beck-view-digitize.exe ..
echo "Executable `beck-view-digitize.exe` ready for usage in directory %CD%"