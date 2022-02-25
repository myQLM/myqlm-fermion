#!/usr/bin/env bash

# This script will disappear and will be replaced by a global script
# written by Jerome (like "bldit")

# Compile
mkdir -p build
cmake -B build -DCMAKE_INSTALL_PREFIX:PATH=$HOME/.local/qlm && cd build && make -j4 install
