// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MoatKitDemo",
    platforms: [
        .macOS(.v15),
    ],
    products: [
        .executable(
            name: "MoatKitDemo",
            targets: ["MoatKitDemo"]),
    ],
    dependencies: [
        .package(path: "../MoatKit"),
    ],
    targets: [
        .executableTarget(
            name: "MoatKitDemo",
            dependencies: ["MoatKit"]),
    ]
)
