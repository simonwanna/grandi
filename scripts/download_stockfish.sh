#!/bin/bash
set -e

# Directory setup
mkdir -p web/bin
TARGET="web/bin/stockfish_engine"

if [ -f "$TARGET" ]; then
    echo "Stockfish already installed at $TARGET"
    exit 0
fi

# Detect OS and Architecture
OS="$(uname -s)"
ARCH="$(uname -m)"
BASE_URL="https://github.com/official-stockfish/Stockfish/releases/download/sf_16"
URL=""
FILE="stockfish_download.tar"
EXTRACTED_DIR=""
BINARY_NAME=""

echo "Detected System: $OS ($ARCH)"

if [ "$OS" = "Linux" ]; then
    # Default to Ubuntu x86-64 for Linux
    URL="$BASE_URL/stockfish-ubuntu-x86-64.tar"
    EXTRACTED_DIR="stockfish-ubuntu-x86-64"
    BINARY_NAME="stockfish-ubuntu-x86-64"

elif [ "$OS" = "Darwin" ]; then
    # MacOS
    if [ "$ARCH" = "arm64" ]; then
        # Apple Silicon (M1/M2/M3)
        URL="$BASE_URL/stockfish-macos-m1-apple-silicon.tar"
        EXTRACTED_DIR="stockfish-macos-m1-apple-silicon"
        BINARY_NAME="stockfish-macos-m1-apple-silicon"
    else
        # Intel Mac
        URL="$BASE_URL/stockfish-macos-x86-64-modern.tar"
        EXTRACTED_DIR="stockfish-macos-x86-64-modern"
        BINARY_NAME="stockfish-macos-x86-64-modern"
    fi
else
    echo "Unsupported Operating System: $OS. Please install Stockfish manually."
    exit 1
fi

echo "Downloading Stockfish from: $URL"
curl -L -o "web/bin/$FILE" "$URL"

echo "Extracting archive..."
tar -xf "web/bin/$FILE" -C "web/bin"

echo "Installing binary to $TARGET..."
# Move binary to standard location
# Note: The extracted folder structure usually matches the tar name
mv "web/bin/$EXTRACTED_DIR/$BINARY_NAME" "$TARGET"
chmod +x "$TARGET"

# Cleanup
rm "web/bin/$FILE"
rm -rf "web/bin/$EXTRACTED_DIR"

echo "Success! Stockfish installed."
