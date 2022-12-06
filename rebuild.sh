rm -rf build
mkdir build
cd build
cmake ../src
make -j2
cp tm_hrnet_timvx ../bin/tm_hrnet_timvx 

