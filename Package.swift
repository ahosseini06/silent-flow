// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "LipReader",
    platforms: [.macOS(.v15)],
    targets: [
        .executableTarget(
            name: "LipReader",
            path: "Sources/LipReader",
            swiftSettings: [.swiftLanguageMode(.v5)]
        )
    ]
)
