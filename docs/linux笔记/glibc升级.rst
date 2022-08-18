=================
glibc升级
=================

centos 升级glibc
======================

glibc版本低，会出现以下错误

::
    
    undefined reference to `lgammaf@GLIBC_2.23

解决方法是升级glibc到2.23版本

1. 安装包下载

.. code-block:: shell

    wget http://mirrors.ustc.edu.cn/gnu/glibc/glibc-2.23.tar.gz


2. 编译安装

.. code-block:: shell

    tar xf glibc-2.23.tar.gz
    cd glibc-2.23/
    mkdir glibc-build
    cd glibc-build #(一定要在新建的目录中操作)
    ../configure --prefix=/usr --disable-sanity-checks --disable-werror 
    make
    make install


在make install 时可能会跳出错误（类似的应该是因为软链接的版本不对造成的）

::

    gawk '/\.gnu\.glibc-stub\./ { \
          sub(/\.gnu\.glibc-stub\./, "", $2); \
          stubs[$2] = 1; } \
        END { for (s in stubs) print "#define __stub_" s }' > /root/glibc-2.23/glibc-build/math/stubsT
    gawk: error while loading shared libraries: /lib64/libm.so.6: invalid ELF header
    make[2]: *** [/root/glibc-2.23/glibc-build/math/stubs] Error 127
    make[2]: Leaving directory `/root/glibc-2.23/math'
    make[1]: *** [math/subdir_install] Error 2
    make[1]: Leaving directory `/root/glibc-2.23'
    make: *** [install] Error 2


解决办法(在另外的窗口执行)：

.. code-block:: shell
    
    cd /lib64
    unlink libm.so.6
    ln -s libm-2.23.so libm.so.6


然后再次执行`make install`
看到如下信息就是安装成功了

::

    LD_SO=ld-linux-x86-64.so.2 CC="gcc" /usr/bin/perl scripts/test-installation.pl /root/glibc-2.23/glibc-build/
    Your new glibc installation seems to be ok.
    make[1]: Leaving directory `/root/glibc-2.23'


3. 验证

:: 
    # ldd --version
    ldd (GNU libc) 2.23
    Copyright (C) 2016 Free Software Foundation, Inc.
    This is free software; see the source for copying conditions.  There is NO
    warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
    Written by Roland McGrath and Ulrich Drepper.
