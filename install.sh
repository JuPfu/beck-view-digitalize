rm -rx build dist
rm -rf *.c *.so
python setup.py build_ext --inplace
mkdir -p dist
mv *.so dist/
pyinstaller beck-view-digitize.spec --noconfirm
mv dist/beck-view-digitize .
chmod +x ./beck-view-digitize