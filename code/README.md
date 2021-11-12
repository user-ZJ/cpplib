# 编译

## linux

```shell
cd code/root/path
mkdir build
cd build
cmake .. #如果需要编译benchmark，则使用cmake -DBUILD_BENCHMARK=ON ..
# cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j 4
```

## Android

```shell
# 编译64位库
export ANDROID_NDK=/opt/env/android-ndk-r22b
mkdir build_android64
cd build_android64
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="arm64-v8a" -DANDROID_NDK=$ANDROID_NDK -DANDROID_PLATFORM=android-21 -DANDROID_STL=c++_static ..
make -j 4

# 编译32位库
export ANDROID_NDK=/opt/env/android-ndk-r22b
mkdir build_android32
cd build_android32
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="armeabi-v7a" -DANDROID_NDK=$ANDROID_NDK -DANDROID_PLATFORM=android-21 -DANDROID_STL=c++_static ..
make -j 4
```

