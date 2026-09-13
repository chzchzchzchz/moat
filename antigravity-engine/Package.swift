// swift-tools-version: 5.9
//
// Project Antigravity — Swift Package Manager (SPM) Manifest
// Distribution package for iOS / macOS Apple Silicon Engine
//

import PackageDescription

let package = Package(
    name: "AntigravityEngine",
    platforms: [
        .iOS(.v16),
        .macOS(.v13)
    ],
    products: [
        .library(
            name: "AntigravityEngine",
            targets: ["AntigravityEngine"]
        ),
    ],
    targets: [
        .binaryTarget(
            name: "CAntigravityEngine",
            path: "frameworks/AntigravityEngine.xcframework"
        ),
        .target(
            name: "AntigravityEngine",
            dependencies: ["CAntigravityEngine"],
            path: "Sources/AntigravityEngine",
            resources: [
                .process("Resources")
            ],
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("Foundation"),
                .linkedLibrary("c++")
            ]
        ),
        .testTarget(
            name: "AntigravityEngineTests",
            dependencies: ["AntigravityEngine"],
            path: "tests/AntigravityEngineTests"
        ),
    ]
)
