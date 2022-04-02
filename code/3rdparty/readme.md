# 第三方依赖软件列表

| 软件 | url | tag | 编译选项 |
| ---- | --- | --- | ------- |
| gflag | https://github.com/gflags/gflags | v2.2.0 | -DCMAKE_CXX_FLAGS=-fPIC |
| glog  | https://github.com/google/glog | v0.5.0 | -DWITH_CUSTOM_PREFIX=ON -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=OFF |
| poco | https://pocoproject.org/releases/poco-1.11.1/poco-1.11.1-all.tar.gz | 1.11.1 | -DBUILD_SHARED_LIBS=OFF |
| fvad | https://github.com/dpirch/libfvad.git | 847a37297a8ca3fe80c4d878a2003f2c5106b0bf | | 
| crypto++ | https://cryptopp.com/cryptopp860.zip | | make -j 4;make install PREFIX=${INSTALL_DIR} |
| minizip | https://github.com/zlib-ng/minizip-ng.git | 99d39015e29703af2612277180ea586805f655ea | |
