cd /d %~dp0

rmdir /S /Q dist
del *.c *.pyd

python3 -m pip install --upgrade pip setuptools wheel cython

python3 setup.py clean --all
python3 setup.py build_ext --inplace

mkdir dist
move *.pyd dist

pyinstaller beck-view-digitize.spec --noconfirm

echo Copying libpng and zlib runtime DLLs...

set TRIPLET=x64-windows

copy "%VCPKG_ROOT%\installed\%TRIPLET%\bin\libpng16.dll" dist\beck-view-digitize-bundle\ >nul
copy "%VCPKG_ROOT%\installed\%TRIPLET%\bin\zlib1.dll" dist\beck-view-digitize-bundle\ >nul

copy /y dist\beck-view-digitize-bundle\beck-view-digitize.exe "%CD%"

echo Executable beck-view-digitize.exe ready for usage in directory %CD%
