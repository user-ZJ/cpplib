==============
gcc升级
==============

centos升级gcc
==============

使用第三方源安装
----------------------

https://www.vpser.net/manage/centos-6-upgrade-gcc.html

.. code-block:: shell

    # 升级到gcc 7.3
    yum -y install centos-release-scl
    yum -y install devtoolset-7-gcc devtoolset-7-gcc-c++ devtoolset-7-binutils
    scl enable devtoolset-7 bash

    # 升级到gcc 8.3
    yum -y install centos-release-scl
    yum -y install devtoolset-8-gcc devtoolset-8-gcc-c++ devtoolset-8-binutils
    scl enable devtoolset-8 bash

    # 升级到gcc 9.3
    yum -y install centos-release-scl
    yum -y install devtoolset-9-gcc devtoolset-9-gcc-c++ devtoolset-9-binutils
    scl enable devtoolset-9 bash

    # 升级到gcc 4.8
    wget http://people.centos.org/tru/devtools-2/devtools-2.repo -O /etc/yum.repos.d/devtoolset-2.repo
    yum -y install devtoolset-2-gcc devtoolset-2-gcc-c++ devtoolset-2-binutils
    scl enable devtoolset-2 bash

    # 升级到gcc4.9
    wget https://copr.fedoraproject.org/coprs/rhscl/devtoolset-3/repo/epel-6/rhscl-devtoolset-3-epel-6.repo -O /etc/yum.repos.d/devtoolset-3.repo
    yum -y install devtoolset-3-gcc devtoolset-3-gcc-c++ devtoolset-3-binutils
    scl enable devtoolset-3 bash

    # 升级到gcc 5.2
    wget https://copr.fedoraproject.org/coprs/hhorak/devtoolset-4-rebuild-bootstrap/repo/epel-6/hhorak-devtoolset-4-rebuild-bootstrap-epel-6.repo -O /etc/yum.repos.d/devtoolset-4.repo
    yum install devtoolset-4-gcc devtoolset-4-gcc-c++ devtoolset-4-binutils -y
    scl enable devtoolset-4 bash


.. note::

    scl命令启用只是临时的，退出shell或重启就会恢复原系统gcc版本。
    如果要长期使用gcc 7.3的话：echo "source /opt/rh/devtoolset-7/enable" >>/etc/profile


源码安装
------------

1. 需要的安装包

* gcc-7.5.0.tar.gz
* gmp-6.1.0.tar.bz2
* mpc-1.0.3.tar.gz
* mpfr-3.1.4.tar.bz2
* isl-0.16.1.tar.bz2

.. code-block:: shell

    wget https://mirrors.ustc.edu.cn/gnu/gcc/gcc-7.5.0/gcc-7.5.0.tar.gz
    wget https://mirrors.ustc.edu.cn/gnu/gmp/gmp-6.1.0.tar.bz2
    wget https://mirrors.ustc.edu.cn/gnu/mpc/mpc-1.0.3.tar.gz
    wget https://mirrors.ustc.edu.cn/gnu/mpfr/mpfr-3.1.4.tar.bz2
    wget http://ftp.ntua.gr/mirror/gnu/gcc/infrastructure/isl-0.16.1.tar.bz2

2. 安装

.. code-block:: shell

    tar -xvf gcc-7.5.0.tar.gz
    cd gcc-7.5.0
    # 把：gmp-6.1.0.tar.bz2、mpc-1.0.3.tar.gz、mpfr-3.1.4.tar.bz2、isl-0.16.1.tar.bz2上传到gcc7.5.0解压目录
    ./contrib/download_prerequisites
    mkdir build
    cd build
    ../configure --enable-checking=release --enable-languages=c,c++ --disable-multilib
    make #（建议不要使用make -j来编译，虽然可以缩短编译时间，但极大可能会编译失败）
    make install

3. 修改基础库链接

**运行程序gcc出现'GLIBCXX_3.4.21' not found**

这是因为升级gcc时，生成的动态库没有替换老版本gcc的动态库导致的，将gcc最新版本的动态库替换系统中老版本的动态库即可解决，运行以下命令检查动态库：

.. code-block:: shell

    rm /lib64/libstdc++.so.6
    ln -s /usr/local/lib64/libstdc++.so.6.0.24 /lib64/libstdc++.so.6
    strings /lib64/libstdc++.so.6 | grep GLIBC


4. 解决cmake没有使用新安装的gcc的问题

cmake执行编译的时候，默认使用/usr/bin目录下的gcc/g++去进行编译，很多时候我们的库文件是用新版本的gcc编译的，所以会出现莫名其妙的错误，
但是错误会包含 gcclib版本之类的信息。

自己装的gcc一般目录在/usr/local/bin目录下，所以需要制定gcc的目录

.. code-block:: shell

    export CC=/usr/local/bin/gcc
    export CXX=/usr/local/bin/g++


ubuntu升级gcc
====================

通过源升级
---------------

.. code-block:: shell

    sudo apt install software-properties-common
    sudo add-apt-repository ppa:ubuntu-toolchain-r/test
    sudo apt update
    sudo apt install gcc-9 g++-9

    # 切换gcc 版本
    sudo apt install gcc-7 g++-7 gcc-8 g++-8 gcc-9 g++-9 gcc-10 g++-10
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100 --slave /usr/bin/g++ g++ /usr/bin/g++-10 --slave /usr/bin/gcov gcov /usr/bin/gcov-10
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 --slave /usr/bin/g++ g++ /usr/bin/g++-9 --slave /usr/bin/gcov gcov /usr/bin/gcov-9
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 80 --slave /usr/bin/g++ g++ /usr/bin/g++-8 --slave /usr/bin/gcov gcov /usr/bin/gcov-8
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 70 --slave /usr/bin/g++ g++ /usr/bin/g++-7 --slave /usr/bin/gcov gcov /usr/bin/gcov-7
    sudo update-alternatives --config gcc

