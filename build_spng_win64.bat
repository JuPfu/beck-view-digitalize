@echo off
setlocal enabledelayedexpansion

:: ==========================================================
::   CONFIG
:: ==========================================================
set BUILD_DIR=%CD%\build
set INSTALL_DIR=%BUILD_DIR%\install
set ARCH=x64

:: ==========================================================
::   PREPARE
:: ==========================================================
echo Creating build directories...
mkdir %BUILD_DIR% 2>nul
mkdir %INSTALL_DIR% 2>nul
mkdir %INSTALL_DIR%\include 2>nul
mkdir %INSTALL_DIR%\lib 2>nul

:: Load MSVC environment
echo Initializing MSBuild environment...
call "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

:: ==========================================================
::   FETCH LIBDEFLATE
:: ==========================================================
cd %BUILD_DIR%
if not exist libdeflate (
    echo Cloning libdeflate...
    git clone https://github.com/ebiggers/libdeflate.git
)
cd libdeflate

echo Configuring libdeflate...
cmake -B build_win -A %ARCH% -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DLIBDEFLATE_BUILD_STATIC_LIB=ON .

echo Building libdeflate...
cmake --build build_win --config Release

echo Installing libdeflate...
copy build_win\Release\libdeflate.lib %INSTALL_DIR%\lib\
xcopy /E /I include %INSTALL_DIR%\include\libdeflate\

:: ==========================================================
::   FETCH LIBSPNG
:: ==========================================================
cd %BUILD_DIR%
if not exist libspng (
    echo Cloning libspng...
    git clone https://github.com/randy408/libspng.git
)
cd libspng

echo Configuring libspng...
cmake -B build_win -A %ARCH% -DCMAKE_BUILD_TYPE=Release ^
    -DSPNG_SHARED=OFF -DSPNG_STATIC=ON ^
    -DSPNG_USE_MINIZ=OFF ^
    -DSPNG_USE_ZLIB=OFF ^
    -DSPNG_USE_LIBDEFLATE=ON ^
    -Dlibdeflate_INCLUDE_DIR=%INSTALL_DIR%\include\libdeflate ^
    -Dlibdeflate_LIBRARY=%INSTALL_DIR%\lib\libdeflate.lib .

echo Building libspng...
cmake --build build_win --config Release

echo Installing libspng...
copy build_win\Release\spng.lib %INSTALL_DIR%\lib\
xcopy /E /I include %INSTALL_DIR%\include\spng\

echo.
echo =====================================================
echo   SUCCESS — libspng + libdeflate built and installed
echo   Include dir: %INSTALL_DIR%\include
echo   Library dir: %INSTALL_DIR%\lib
echo =====================================================
echo.

endlocal
