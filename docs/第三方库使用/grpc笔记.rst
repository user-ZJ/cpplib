======================================
grpc 笔记
======================================

安装
==================

编译源码
------------

.. code-block:: shell

    export MY_INSTALL_DIR=$HOME/.local
    mkdir -p $MY_INSTALL_DIR
    export PATH="$MY_INSTALL_DIR/bin:$PATH"
    # cmake 要求3.13以上
    wget -q -O cmake-linux.sh https://github.com/Kitware/CMake/releases/download/v3.19.6/cmake-3.19.6-Linux-x86_64.sh
    sh cmake-linux.sh -- --skip-license --prefix=$MY_INSTALL_DIR
    # 安装依赖库
    sudo apt install -y build-essential autoconf libtool pkg-config
    # 下载源码
    git clone --recursive -b v1.49.x https://github.com/grpc/grpc.git
    # 编译
    cd grpc
    mkdir -p cmake/build
    pushd cmake/build
    cmake -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=$MY_INSTALL_DIR ../..
    make -j 4
    make install
    popd


编译helloworld
----------------------

.. code-block:: shell

    cd examples/cpp/helloworld
    mkdir -p cmake/build
    pushd cmake/build
    cmake -DCMAKE_PREFIX_PATH=$MY_INSTALL_DIR ../..
    make -j
    # 运行
    ./greeter_server &
    ./greeter_client

编译route_guide
-------------------------------

.. code-block:: shell

    cd examples/cpp/route_guide
    mkdir -p cmake/build
    cd cmake/build
    cmake -DCMAKE_PREFIX_PATH=$MY_INSTALL_DIR ../..
    make -j 4


参考
=============
https://grpc.io/docs/languages/cpp/quickstart/

