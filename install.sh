rm -rf dist
rm -rf *.c *.so
python -m pip install --upgrade pip setuptools wheel cython
python setup.py clean -a
python setup.py build_ext -i
mkdir -p dist
mv *.so dist/
pyinstaller beck-view-digitize.spec --noconfirm
mv dist/beck-view-digitize .
chmod +x ./beck-view-digitize
dir=$(pwd -P)
echo 'Executable `beck-view-digitize` ready for usage in directory' $dir