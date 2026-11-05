// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "SilentFlow",
    platforms: [.macOS(.v15)],
    targets: [
        .executableTarget(
            name: "SilentFlow",
            path: "Sources/SilentFlow",
            swiftSettings: [.swiftLanguageMode(.v5)]
        )
    ]
)
