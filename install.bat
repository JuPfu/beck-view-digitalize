cd /d %~dp0

rmdir /S /Q dist
del *.c *.pyd

python3 -m pip install --upgrade pip setuptools wheel cython

python3 setup.py clean --all
python3 setup.py build_ext --inplace

mkdir dist
move *.pyd dist

pyinstaller beck-view-digitize.spec --noconfirm

copy /y dist\beck-view-digitize-bundle\beck-view-digitize.exe "%CD%"

echo Executable beck-view-digitize.exe ready for usage in directory %CD%
