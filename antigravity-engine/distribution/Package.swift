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
            checksum: "a19b71534d1dea54bd1618b8c6e91244cb642528de6eff0e70c2c3ac624185e9"
        )
    ]
)
