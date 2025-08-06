rmdir build
rmdir dist
del *.c *.pyd
python setup.py build_ext --inplace
mkdir dist
move build\lib.win-amd64-cpython-313\*.pyd dist
pyinstaller beck-view-digitize.spec --noconfirm
move /y dist\beck-view-digitize-bundle\beck-view-digitize.exe .
echo "Executable `beck-view-digitize.exe` ready for use in %CD%"