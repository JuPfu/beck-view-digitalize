rm -rf dist
rm -rf *.c *.so
python setup.py clean --all
python setup.py build_ext --inplace
mkdir -p dist
mv *.so dist/
pyinstaller beck-view-digitize.spec --noconfirm
mv dist/beck-view-digitize .
chmod +x ./beck-view-digitize
dir=$(pwd -P)
echo 'Executable `beck-view-digitize` ready for usage in directory' $dir