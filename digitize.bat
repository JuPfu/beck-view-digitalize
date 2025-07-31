@echo on
rem This script was created by Nuitka to execute 'beck-view-digitize.exe' with Python DLL being found.
set PATH="C:\Users\Peter lokal\AppData\Local\Programs\Python\Python313";%PATH%
set PYTHONHOME="C:\Users\Peter lokal\AppData\Local\Programs\Python\Python313"
cd "%~dp0"
call venv\Scripts\activate.bat
beck-view-digitize %*
deactivate