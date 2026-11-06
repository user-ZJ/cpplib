#!/usr/bin/env bash
# Explicit dependency installation, separate from project configure/build.
set -euo pipefail
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
prefix="${MBASE_DEPS_PREFIX:-$repo_dir/.local-deps}"
work="${MBASE_DEPS_WORK:-$repo_dir/.deps-work}"
jobs="${JOBS:-4}"
cache="${MBASE_SOURCE_CACHE:-}"
if [[ "${1:-}" == --help ]]; then
    echo 'Usage: scripts/install_deps.sh [--help]'
    echo 'INSTALL_ONNX=1: also install ONNX Runtime; INSTALL_KALDI=0: skip compatible OpenFST.'
    echo 'MBASE_DEPS_PREFIX: install prefix; JOBS: build jobs; MBASE_SOURCE_CACHE: optional existing source cache (fc_base).'
    echo 'Requires: CMake, C/C++ compiler, make, curl, tar, unzip, OpenSSL, sox and sonic development files.'
    exit 0
fi
[[ $# == 0 ]] || { echo 'Unknown argument' >&2; exit 2; }
mkdir -p "$prefix" "$work"
export CMAKE_PREFIX_PATH="$prefix${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
if command -v brew >/dev/null; then
    export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:$(brew --prefix)"
fi
fetch() {
    local name="$1" url="$2" hash="$3"
    src="$work/$name-src"
    if [[ ! -d "$src" ]]; then
        if [[ -n "$cache" && -d "$cache/$name-src" ]]; then
            cp -R "$cache/$name-src" "$src"
        else
            curl --fail --location --retry 3 "$url" -o "$work/$name.archive"
            echo "$hash  $work/$name.archive" | shasum -a 256 -c -
            mkdir -p "$src"
            if [[ "$url" == *.zip ]]; then
                mkdir -p "$work/$name-unpack"
                unzip -q "$work/$name.archive" -d "$work/$name-unpack"
                cp -R "$work/$name-unpack/"*/. "$src/"
            else
                tar -xf "$work/$name.archive" -C "$src" --strip-components=1
            fi
        fi
    fi
}
build() {
    cmake -S "$src" -B "$work/$1-build" -DCMAKE_INSTALL_PREFIX="$prefix" \
        -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_EXPORT_NO_PACKAGE_REGISTRY=ON -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_POSITION_INDEPENDENT_CODE=ON "${@:2}"
    cmake --build "$work/$1-build" --parallel "$jobs"
    cmake --install "$work/$1-build"
}
fetch gflags https://github.com/gflags/gflags/archive/refs/tags/v2.3.1.zip d08fb50ffe6fffb7e895f72ef0c160059fd88a676978bd20c69f2299384c8e03
build gflags -DREGISTER_INSTALL_PREFIX=OFF -DREGISTER_BUILD_DIR=OFF -DBUILD_TESTING=OFF -DGFLAGS_BUILD_TESTING=OFF -DGFLAGS_BUILD_SHARED_LIBS=ON
fetch glog https://github.com/google/glog/archive/refs/tags/v0.7.1.zip c17d85c03ad9630006ef32c7be7c65656aba2e7e2fbfc82226b7e680c771fc88
build glog -DBUILD_TESTING=OFF
fetch poco https://github.com/pocoproject/poco/archive/refs/tags/poco-1.14.2-release.tar.gz 47394ea7ddb7b0a40e1a5be896f8f5dc77cfdc4f561d2e7131ecf582df5a0c3a
cmake -DPOCO_SOURCE_DIR="$src" -P "$repo_dir/3rdparty/patches/fix_poco.cmake"
build poco -DENABLE_TESTS=OFF -DENABLE_SAMPLES=OFF -DENABLE_DATA_MYSQL=OFF -DENABLE_DATA_ODBC=OFF -DENABLE_DATA_POSTGRESQL=OFF
fetch googletest https://github.com/google/googletest/archive/refs/tags/release-1.11.0.tar.gz b4870bf121ff7795ba20d20bcdd8627b8e088f2d1dab299a031c1034eddc93d5
build googletest -DBUILD_GMOCK=OFF
fetch rnnoise https://github.com/xiph/rnnoise/archive/7f449bf8bd3b933891d12c30112268c4090e4d59.tar.gz 05396f221b308099491d9f4838b58a71ba2224a5e4a1c5276de6385abf369131
cp "$repo_dir/3rdparty/patches/rnnoise/CMakeLists.txt" "$src/CMakeLists.txt"
build rnnoise
fetch boost https://archives.boost.io/release/1.87.0/source/boost_1_87_0.tar.gz f55c340aa49763b1925ccf02b2e83f35fdcf634c9d5164a2acb87540173c741d
mkdir -p "$prefix/include"
cp -R "$src/boost" "$prefix/include/"
fetch rapidjson https://github.com/Tencent/rapidjson/archive/refs/tags/v1.1.0.tar.gz bf7ced29704a1e696fbccf2a2b4ea068e7774fa37f6d7dd4039d0787f8bed98e
cp -R "$src/include/." "$prefix/include/"
fetch eigen https://gitlab.com/libeigen/eigen/-/archive/3.4.1/eigen-3.4.1.tar.gz b93c667d1b69265cdb4d9f30ec21f8facbbe8b307cf34c0b9942834c6d4fdbe2
mkdir -p "$prefix/include/eigen3"
cp -R "$src/Eigen" "$src/unsupported" "$prefix/include/eigen3/"
fetch cereal https://github.com/user-ZJ/cereal/archive/refs/tags/v1.3.2-beta.5.zip fedfab6ae9248ebd7c57744c2831219435a3b6bf499a8da9131b84a12490a3c1
cp -R "$src/include/." "$prefix/include/"
if [[ "${INSTALL_ONNX:-0}" == 1 ]]; then
    case "$(uname -s)-$(uname -m)" in
        Darwin-arm64) platform=osx-arm64; hash=23117b6f5d7324d4a7c51184e5f808dd952aec411a6b99a1b6fd1011de06e300 ;;
        Darwin-x86_64) platform=osx-x86_64; hash=09b17f712f8c6f19bb63da35d508815b443cbb473e16c6192abfaa297c02f600 ;;
        Linux-x86_64) platform=linux-x64; hash=5d503ce8540358b59be26c675e42081be14a3e833a5301926f555451046929c5 ;;
        Linux-aarch64) platform=linux-aarch64; hash=5820d9f343df73c63b6b2b174a1ff62575032e171c9564bcf92060f46827d0ac ;;
        *) echo 'Unsupported ONNX Runtime platform' >&2; exit 1 ;;
    esac
    fetch onnxruntime "https://github.com/microsoft/onnxruntime/releases/download/v1.12.0/onnxruntime-$platform-1.12.0.tgz" "$hash"
    cp -R "$src/include/." "$prefix/include/"
    mkdir -p "$prefix/lib"
    cp -R "$src/lib/." "$prefix/lib/"
fi
if [[ "${INSTALL_KALDI:-1}" == 1 ]]; then
    fetch openfst https://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.6.5.tar.gz 70d9e795b9bb3eb157908a2fdde17d5d1c1be35ab21e208187a4f230291606b1
    (cd "$src" && cmake -P "$repo_dir/3rdparty/patches/fix_openfst.cmake")
    mkdir -p "$work/openfst-build"
    (cd "$work/openfst-build"
     "$src/configure" --prefix="$prefix" --disable-bin --enable-static --disable-shared \
        CXXFLAGS="-O2 -fPIC" CPPFLAGS="-I$prefix/include -DGLOG_USE_GLOG_EXPORT" \
        LDFLAGS="-L$prefix/lib -Wl,-rpath,$prefix/lib" LIBS="-lgflags -lglog -lpthread"
     make -j "$jobs"
     make install)
fi
echo "Dependencies installed in $prefix"
echo "Configure: cmake -S . -B build-local -DMBASE_DEPS_PREFIX=$prefix -DUSE_ONNXRUNTIME=OFF"
