import Foundation

// `LipReader --probe <video> <out-dir>` runs the exact face tracking, mouth
// cropping and segmentation code the live app uses, on a video file, and
// writes the 25 fps mouth crops + detected utterances for offline checking.
let args = CommandLine.arguments
if args.count >= 3, args[1] == "--render-onboarding" {
    exit(MainActor.assumeIsolated { OnboardingSnapshots.render(to: URL(fileURLWithPath: args[2])) })
} else if args.count >= 4, args[1] == "--probe" {
    exit(Probe.run(video: URL(fileURLWithPath: args[2]), outDir: URL(fileURLWithPath: args[3])))
} else {
    LipReaderApp.main()
}
