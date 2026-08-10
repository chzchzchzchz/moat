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
        .target(
            name: "CAntigravityEngine",
            publicHeadersPath: "include",
            cSettings: [
                .headerSearchPath("include")
            ]
        ),
        .target(
            name: "AntigravityEngine",
            dependencies: ["CAntigravityEngine"],
            path: "Sources/AntigravityEngine"
        ),
        .testTarget(
            name: "AntigravityEngineTests",
            dependencies: ["AntigravityEngine"],
            path: "Tests/AntigravityEngineTests"
        ),
    ]
)
