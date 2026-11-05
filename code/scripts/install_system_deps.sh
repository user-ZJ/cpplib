#!/usr/bin/env bash
# Run explicitly; may request sudo on Debian/Ubuntu.
set -euo pipefail
case "${1:-}" in --help) echo 'Install system build tools, OpenSSL, sox, sonic and gRPC (macOS or Debian/Ubuntu).'; exit 0 ;; '') ;; *) exit 2 ;; esac
case "$(uname -s)" in
 Darwin)
    command -v brew >/dev/null || { echo 'Install Homebrew first: https://brew.sh' >&2; exit 1; }
    brew install cmake pkg-config openssl@3 sox opus grpc protobuf
    if [[ ! -f "$(brew --prefix)/include/sonic.h" ]]; then
        work_dir="$(mktemp -d)"
        git clone --depth 1 https://github.com/waywardgeek/sonic.git "$work_dir/sonic"
        make -C "$work_dir/sonic" PREFIX="$(brew --prefix)"
        make -C "$work_dir/sonic" PREFIX="$(brew --prefix)" install
        echo "Sonic source retained at $work_dir/sonic"
    fi
    ;;
 Linux)
    command -v apt-get >/dev/null || { echo 'Install equivalent development packages with your package manager; see README.md.' >&2; exit 1; }
    sudo apt-get update
    sudo apt-get install -y build-essential cmake pkg-config curl unzip git libssl-dev libsox-dev libsonic-dev libopus-dev libgrpc++-dev libprotobuf-dev protobuf-compiler protobuf-compiler-grpc
    ;;
 *) echo 'Unsupported platform; install dependencies manually.' >&2; exit 1 ;;
esac
