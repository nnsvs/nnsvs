
# install hts_engine_API
pushd hts_engine_API/src
touch ChangeLog
export CFLAGS="$CFLAGS -fPIC"
autoreconf -fvi
./configure
make clean
make
sudo make install
popd

# install bandmat
pushd bandmat
python3 setup.py install
popd



# install nnmnkwii
if [ "`pip freeze | grep nnmnkwikk`" != '' ]; then
    pip uninstall nnmnkwii
fi
pushd nnmnkwii
python3 setup.py install
popd

# install sinsy
pushd sinsy/src
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON  ..
make -j
sudo make install
popd

# install pysinsy
pushd pysinsy
python3 setup.py develop
popd

# set enviromental variable
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
