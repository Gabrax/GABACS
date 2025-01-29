#!/bin/bash

OS=$(uname)
EXE_NAME="AntiCollision"
EXE_EXTENSION=""
EXE_PATH="$EXE_NAME$EXE_EXTENSION"

RED="\e[31m"
GREEN="\e[32m"
YELLOW="\e[33m"
RESET="\e[0m"

# Set the compiler based on the argument
COMPILER=$1
RELEASE_MODE=false

if [[ "$2" == "release" ]]; then
    RELEASE_MODE=true
fi

if [[ "$OS" == "Linux" || "$OS" == "Darwin" ]]; then
    # Linux or macOS
    EXE_EXTENSION=""
    FIND_CMD="find . -type f -name \"$EXE_NAME\" -print -quit"
    echo -e "${YELLOW}[*] Running on Linux or macOS${RESET}"
elif [[ "$OS" == "MINGW"* || "$OS" == "CYGWIN"* || "$OS" == "MSYS"* ]]; then
    # Windows (using Git Bash or similar)
    EXE_EXTENSION=".exe"
    FIND_CMD="find . -type f -name \"$EXE_NAME$EXE_EXTENSION\" -print -quit"
    echo -e "${YELLOW}[*] Running on Windows${RESET}"
else
    echo -e "${RED}Unsupported OS: $OS${RESET}"
    exit 1
fi

BUILD_DIR="build"
ROOT_DIR=$(pwd)

# Check if vcpkg folder exists
if [ ! -d "vcpkg" ]; then
    echo -e "${YELLOW}[*] vcpkg folder does not exist. Cloning vcpkg...${RESET}"
    git clone https://github.com/microsoft/vcpkg.git || { echo -e "${RED}[*] Failed to clone vcpkg${RESET}"; exit 1; }
fi

cd vcpkg || { echo -e "${RED}[*] Failed to navigate to vcpkg folder${RESET}"; exit 1; }

# Bootstrap vcpkg
if [[ "$OS" == "Linux" || "$OS" == "Darwin" ]]; then
    echo -e "${YELLOW}[*] Bootstrapping vcpkg for Linux or macOS${RESET}"
    ./bootstrap-vcpkg.sh || { echo -e "${RED}[*] Failed to bootstrap vcpkg${RESET}"; exit 1; }
elif [[ "$OS" == "MINGW"* || "$OS" == "CYGWIN"* || "$OS" == "MSYS"* ]]; then
    echo -e "${YELLOW}[*] Bootstrapping vcpkg for Windows${RESET}"
    ./bootstrap-vcpkg.bat || { echo -e "${RED}[*] Failed to bootstrap vcpkg${RESET}"; exit 1; }
else
    echo -e "${RED}[*] Unsupported OS for vcpkg bootstrap${RESET}"
    exit 1
fi

echo -e "${YELLOW}[*] Installing OpenCV${RESET}"
./vcpkg.exe integrate install
./vcpkg.exe install opencv4:x64-windows --recurse || { echo -e "${RED}[*] Failed to install OpenCV${RESET}"; exit 1; }


cd "$ROOT_DIR" || { echo -e "${RED}[*] Failed to return to root directory${RESET}"; exit 1; }

# Check if build directory exists
if [ -d "$BUILD_DIR" ]; then
    echo -e "${GREEN}[*] Build directory exists${RESET}"
else
    echo -e "${YELLOW}[*] Build directory does not exist. Creating it now...${RESET}"
    mkdir -p "$BUILD_DIR"
fi

cd "$BUILD_DIR" || { echo -e "${RED}[*] Failed to navigate to build directory${RESET}"; exit 1; }

# Check if project is already configured
if [ -f "CMakeCache.txt" ]; then
    echo -e "${GREEN}[*] Project is already configured${RESET}"
else
    echo -e "${YELLOW}[*] Configuring the project with CMake...${RESET}"
   
    CMAKE_TOOLCHAIN="-DCMAKE_TOOLCHAIN_FILE="../vcpkg/scripts/buildsystems/vcpkg.cmake""
    CMAKE_ARGS=""
    if $RELEASE_MODE; then
        CMAKE_ARGS=" -DRELEASE=ON"
    fi

    if [ -z "$COMPILER" ]; then
        cmake .. $CMAKE_ARGS || { echo -e "${RED}[*] CMake configuration failed${RESET}"; exit 1; }
    elif [ "$COMPILER" == "msvc" ]; then
        echo -e "${YELLOW}[*] Using MSVC Compiler${RESET}"
        cmake -G "Visual Studio 17 2022" -G "Ninja" .. $CMAKE_TOOLCHAIN $CMAKE_ARGS || { echo -e "${RED}[*] CMake configuration failed${RESET}"; exit 1; }
    elif [ "$COMPILER" == "clang" ]; then
        echo -e "${YELLOW}[*] Using Clang Compiler${RESET}"
        cmake -G "Ninja" -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -Wno-dev .. $CMAKE_TOOLCHAIN $CMAKE_ARGS|| { echo -e "${RED}[*] CMake configuration failed${RESET}"; exit 1; }
    elif [ "$COMPILER" == "gcc" ]; then
        echo -e "${YELLOW}[*] Using GCC Compiler${RESET}"
        cmake -G "Unix Makefiles" -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -Wno-dev .. $CMAKE_TOOLCHAIN $CMAKE_ARGS|| { echo -e "${RED}[*] CMake configuration failed${RESET}"; exit 1; }
    else
        echo -e "${RED}[*] Unsupported compiler: $COMPILER${RESET}"
        exit 1
    fi
fi

# Build the project
echo -e "${YELLOW}[*] Building the project...${RESET}"
cmake --build . --config Release || { echo -e "${RED}[*] Build failed${RESET}"; exit 1; }
echo -e "${GREEN}[*] Build completed successfully${RESET}"

# Search for the executable in the build directory
echo -e "${YELLOW}[*] Searching for $EXE_PATH in the build directory...${RESET}"
if mygame_path=$(eval "$FIND_CMD"); then
    echo -e "${GREEN}[*] Executable $EXE_PATH found at: $mygame_path${RESET}"
    
    echo -e "${YELLOW}[*] Running $EXE_PATH${RESET}"
    ./$EXE_PATH || { echo -e "${RED}[*] Failed to run $EXE_PATH${RESET}"; exit 1; }
else
    echo -e "${RED}[*] Executable $EXE_PATH not found${RESET}"
fi



