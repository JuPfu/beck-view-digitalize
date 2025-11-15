rmdir /S /Q dist
del *.c *.pyd
python setup.py clean --all
python setup.py build_ext --inplace
mkdir dist
move *.pyd dist
pyinstaller beck-view-digitize.spec --noconfirm
copy /y dist\beck-view-digitize-bundle\beck-view-digitize.exe "%CD%"
echo "Executable `beck-view-digitize.exe` ready for usage in directory %CD%"