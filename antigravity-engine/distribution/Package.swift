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
            checksum: "fc01fefadc87d7bcea26b03a0b770bd1711b6da4bf0199ed7a3d9bddce14324d"
        )
    ]
)
