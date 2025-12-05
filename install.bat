@echo off
setlocal EnableDelayedExpansion

echo ---------------------------------------
echo Cleaning old builds ...
echo ---------------------------------------
rmdir /S /Q build 2>nul
rmdir /S /Q dist 2>nul
del /Q *.c *.pyd 2>nul

echo ---------------------------------------
echo Upgrading build dependencies ...
echo ---------------------------------------
python -m pip install --upgrade pip setuptools wheel cython

echo ---------------------------------------
echo Running Cython build (static linking) ...
echo ---------------------------------------
python setup.py clean --all
python setup.py build_ext --inplace

echo ---------------------------------------
echo Searching for generated .pyd files ...
echo ---------------------------------------
set pyd_files=
for %%f in (*.pyd) do (
    set pyd_files=!pyd_files! %%f
)

echo Found: !pyd_files!

if "!pyd_files!"=="" (
    echo ERROR: No .pyd files were generated!
    echo Setup.py probably didn't detect your .pyx sources.
    exit /b 1
)

echo ---------------------------------------
echo Moving .pyd files into dist folder ...
echo ---------------------------------------
mkdir dist
for %%f in (!pyd_files!) do (
    move %%f dist >nul
)

echo ---------------------------------------
echo Running PyInstaller ...
echo ---------------------------------------
pyinstaller beck-view-digitize.spec --noconfirm

echo ---------------------------------------
echo Copy final EXE back to project root ...
echo ---------------------------------------
copy /y dist\beck-view-digitize-bundle\beck-view-digitize.exe "%CD%" >nul

echo.
echo #############################################################
echo   Static EXE build complete — NO DLLs required at runtime!
echo #############################################################
echo.
