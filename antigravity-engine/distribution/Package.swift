// swift-tools-version: 5.9
//
// Project Antigravity — Public SPM Binary Distribution Package
// Clean closed-source binary distribution for enterprise consumers.
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
            name: "AntigravityEngine",
            url: "https://github.com/chzchzchzchz/moat/releases/download/v1.0.0/AntigravityEngine.xcframework.zip",
            checksum: "0b402e11c9e28df82a36e0beb16e8d0beae4d06a34420ad5c7e11248b75b15b6"
        )
    ]
)
