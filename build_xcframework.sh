#!/bin/bash
set -e

echo "🚀 Building AntigravityEngineCore.xcframework for iOS and Simulator..."

mkdir -p build_xcf/ios-arm64
mkdir -p build_xcf/ios-simulator

# 1. Build for iOS Device (arm64)
echo "Building for iOS Device (arm64)..."
xcrun -sdk iphoneos clang++ -arch arm64 -O3 -std=c++17 -x objective-c++ -fobjc-arc \
    -c antigravity-engine/src/transformer_engine.mm -o build_xcf/ios-arm64/transformer_engine.o \
    -target arm64-apple-ios16.0
xcrun -sdk iphoneos clang++ -arch arm64 -O3 -std=c++17 -x objective-c++ -fobjc-arc \
    -c antigravity-engine/src/antigravity_c_api.cpp -o build_xcf/ios-arm64/antigravity_c_api.o \
    -target arm64-apple-ios16.0

xcrun -sdk iphoneos ar rcs build_xcf/ios-arm64/libAntigravityEngineCore.a \
    build_xcf/ios-arm64/transformer_engine.o build_xcf/ios-arm64/antigravity_c_api.o

# 2. Build for iOS Simulator (arm64, x86_64)
echo "Building for iOS Simulator (arm64)..."
xcrun -sdk iphonesimulator clang++ -arch arm64 -O3 -std=c++17 -x objective-c++ -fobjc-arc \
    -c antigravity-engine/src/transformer_engine.mm -o build_xcf/ios-simulator/transformer_engine_arm64.o \
    -target arm64-apple-ios16.0-simulator
xcrun -sdk iphonesimulator clang++ -arch arm64 -O3 -std=c++17 -x objective-c++ -fobjc-arc \
    -c antigravity-engine/src/antigravity_c_api.cpp -o build_xcf/ios-simulator/antigravity_c_api_arm64.o \
    -target arm64-apple-ios16.0-simulator

xcrun -sdk iphonesimulator ar rcs build_xcf/ios-simulator/libAntigravityEngineCore_arm64.a \
    build_xcf/ios-simulator/transformer_engine_arm64.o build_xcf/ios-simulator/antigravity_c_api_arm64.o

echo "Building for iOS Simulator (x86_64)..."
xcrun -sdk iphonesimulator clang++ -arch x86_64 -O3 -std=c++17 -x objective-c++ -fobjc-arc \
    -c antigravity-engine/src/transformer_engine.mm -o build_xcf/ios-simulator/transformer_engine_x86_64.o \
    -target x86_64-apple-ios16.0-simulator
xcrun -sdk iphonesimulator clang++ -arch x86_64 -O3 -std=c++17 -x objective-c++ -fobjc-arc \
    -c antigravity-engine/src/antigravity_c_api.cpp -o build_xcf/ios-simulator/antigravity_c_api_x86_64.o \
    -target x86_64-apple-ios16.0-simulator

xcrun -sdk iphonesimulator ar rcs build_xcf/ios-simulator/libAntigravityEngineCore_x86_64.a \
    build_xcf/ios-simulator/transformer_engine_x86_64.o build_xcf/ios-simulator/antigravity_c_api_x86_64.o

# Universal Simulator Lib
xcrun -sdk iphonesimulator lipo -create \
    build_xcf/ios-simulator/libAntigravityEngineCore_arm64.a \
    build_xcf/ios-simulator/libAntigravityEngineCore_x86_64.a \
    -output build_xcf/ios-simulator/libAntigravityEngineCore.a

# 3. Create Header Directory
mkdir -p build_xcf/include
cp antigravity-engine/src/antigravity_c_api.h build_xcf/include/
cp antigravity-engine/src/transformer_engine.h build_xcf/include/

# 4. Create XCFramework
echo "Packaging into AntigravityEngineCore.xcframework..."
rm -rf AntigravityEngineCore.xcframework
xcodebuild -create-xcframework \
    -library build_xcf/ios-arm64/libAntigravityEngineCore.a -headers build_xcf/include \
    -library build_xcf/ios-simulator/libAntigravityEngineCore.a -headers build_xcf/include \
    -output AntigravityEngineCore.xcframework

echo "✅ Successfully built AntigravityEngineCore.xcframework!"
echo "You can now drag and drop AntigravityEngineCore.xcframework and src/AntigravityEngine.swift into your iOS Xcode Project."
