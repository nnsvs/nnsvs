# install hts_engine_API
cd hts_engine_API/src && ./waf configure build && sudo ./waf install
cd ../../ && rm -rf hts_engine_API

# install bandmat
cd bandmat && python setup.py install
cd ../ && rm -rf bandmat

# install nnmnkwii
if [ "`pip freeze | grep nnmnkwii`" != '' ]; then
    pip uninstall nnmnkwii
fi
cd nnmnkwii && python setup.py install
cd ../ && rm -rf nnmnkwii

# install sinsy
cd sinsy/src/ && mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON  .. && make -j && sudo make install
cd ../../ && rm -rf sinsy

# install pysinsy
cd pysinsy && python setup.py develop
cd ../ && rm -rf pysinsy

# set enviromental variable
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
