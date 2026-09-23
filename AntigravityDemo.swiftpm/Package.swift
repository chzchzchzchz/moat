// swift-tools-version: 5.8

import PackageDescription

let package = Package(
    name: "AntigravityDemo",
    platforms: [
        .iOS(.v16), .macOS(.v13)
    ],
    products: [
        .iOSApplication(
            name: "AntigravityDemo",
            targets: ["AppModule"],
            displayVersion: "1.0",
            bundleVersion: "1",
            appIcon: .placeholder(icon: .rocket),
            accentColor: .presetColor(.blue),
            supportedDeviceFamilies: [
                .pad,
                .phone
            ],
            supportedInterfaceOrientations: [
                .portrait,
                .landscapeRight,
                .landscapeLeft,
                .portraitUpsideDown(.when(deviceFamilies: [.pad]))
            ]
            // No `capabilities:` entry: this demo makes no network connections, and the
            // project's central claim is zero network egress. Declaring
            // .outgoingNetworkConnections() requested an entitlement nothing here uses.
        ),
        .executable(name: "AntigravityDemoCLI", targets: ["AppModule"])
    ],
    targets: [
        .target(
            name: "AntigravityCore",
            path: "Core",
            publicHeadersPath: "include",
            cxxSettings: [
                .unsafeFlags(["-O3", "-std=c++17", "-fobjc-arc"])
            ]
        ),
        .executableTarget(
            name: "AppModule",
            dependencies: ["AntigravityCore"],
            path: "App"
        )
    ],
    cxxLanguageStandard: .cxx17
)
