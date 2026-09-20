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
            checksum: "602c86e541f8b0284722f59e9dfdce594eabf06bdb5e0bbf3b6bf7d162fda337"
        )
    ]
)
