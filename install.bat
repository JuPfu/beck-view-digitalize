rmdir build
rmdir dist
del *.c *.so
python setup.py build_ext --inplace
mkdir dist
move *.so dist
pyinstaller beck-view-digitize.spec --noconfirm
move /y dist\beck-view-digitize.exe %~pi..
echo "Executable `beck-view-digitize.exe` ready for usage in directory $(%~pi)"