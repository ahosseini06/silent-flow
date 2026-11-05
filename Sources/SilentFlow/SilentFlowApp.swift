import AppKit
import SwiftUI


struct SilentFlowApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    @ObservedObject private var model = AppModel.shared

    var body: some Scene {
        Window("SilentFlow", id: "main") {
            RootView()
                .environmentObject(model)
                .frame(minWidth: 560, minHeight: 380)
                .preferredColorScheme(.dark)
        }
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 1080, height: 700)

        MenuBarExtra {
            MenuBarContent().environmentObject(model)
        } label: {
            Image(nsImage: LogoMark.menuBarImage(active: model.dictation != nil))
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
        AppModel.shared.start()
    }

    // Keeps running as a push-to-talk "keyboard" from the menu bar when the window is closed.
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }

    func applicationWillTerminate(_ notification: Notification) {
        AppModel.shared.shutdown()
    }
}

/// Onboarding until it's done, then the live camera view. The onboarding lifts
/// away (blur + fade) to reveal the app underneath.
struct RootView: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        ZStack {
            if model.onboarded && (model.calibration != nil || model.calibrateWhenReady) {
                CalibrationView()
                    .transition(.modifier(active: BlurShift(blur: 16, opacity: 0, x: 0),
                                          identity: BlurShift(blur: 0, opacity: 1, x: 0)))
            } else if model.onboarded && model.tutorialActive {
                TutorialView()
                    .transition(.modifier(active: BlurShift(blur: 16, opacity: 0, x: 0),
                                          identity: BlurShift(blur: 0, opacity: 1, x: 0)))
            } else if model.onboarded {
                ContentView()
                    .transition(.opacity)
            } else {
                OnboardingView()
                    .transition(.asymmetric(insertion: .opacity,
                                            removal: .modifier(active: BlurShift(blur: 22, opacity: 0, x: 0),
                                                               identity: BlurShift(blur: 0, opacity: 1, x: 0))
                                                .combined(with: .scale(scale: 1.04))))
            }
        }
        .background(Theme.bg)
        .animation(.easeInOut(duration: 0.7), value: model.calibration == nil)
    }
}

struct ContentView: View {
    @EnvironmentObject var model: AppModel
    @State private var showHint = true

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            CameraPreview(session: model.camera.session)
                .scaleEffect(x: -1, y: 1)  // mirror, like a mirror
                .ignoresSafeArea()

            // Soft scrim so captions stay readable on bright backgrounds.
            LinearGradient(colors: [.clear, .black.opacity(0.55)], startPoint: .center, endPoint: .bottom)
                .ignoresSafeArea()
                .allowsHitTesting(false)

            VStack(spacing: 0) {
                HStack(alignment: .top) {
                    StatusPill()
                    Spacer()
                    if model.showDebug { DebugPanel() }
                }
                .padding(.horizontal, 20)
                .padding(.top, 38)

                if model.calibration == nil && model.phase == .ready {
                    if !model.accessibilityTrusted {
                        AccessibilityNudge().padding(.top, 10).transition(.opacity)
                    } else if !model.personalized {
                        CalibrateNudge().padding(.top, 10).transition(.opacity)
                    }
                }

                Spacer()

                Subtitles()
                    .padding(.horizontal, 48)
                    .padding(.bottom, showHint ? 14 : 40)

                if showHint && model.calibration == nil {
                    Text("Hold \(model.pttKey.short) in any app to type by lip-reading  ·  Space reads here  ·  K calibrate  ·  L context fix  ·  D debug  ·  C clear")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(.white.opacity(0.5))
                        .padding(.bottom, 16)
                        .transition(.opacity)
                }
            }
        }
        .animation(.easeOut(duration: 0.25), value: model.calibration)
        .task {
            try? await Task.sleep(for: .seconds(15))
            withAnimation(.easeOut(duration: 0.6)) { showHint = false }
        }
    }
}

struct StatusPill: View {
    @EnvironmentObject var model: AppModel

    private var state: (String, Color, Bool) {
        switch model.phase {
        case .starting: return ("Starting camera…", .gray, false)
        case .loadingModel: return ("Loading lip-reading model…", .gray, true)
        case .error(let msg): return (msg, .red, false)
        case .ready:
            if model.manualHeld { return ("Reading while you hold the key", .red, true) }
            if !model.faceVisible { return ("Looking for your face…", .orange, false) }
            if model.speaking { return ("Reading lips…", .red, true) }
            return ("Watching", .green, false)
        }
    }

    var body: some View {
        let (text, color, pulse) = state
        HStack(spacing: 8) {
            Circle()
                .fill(color)
                .frame(width: 8, height: 8)
                .opacity(pulse ? 0.55 : 1)
                .animation(pulse ? .easeInOut(duration: 0.6).repeatForever() : .default, value: pulse)
            Text(text)
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(.white.opacity(0.9))
                .lineLimit(2)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 7)
        .background(.black.opacity(0.45), in: Capsule())
        .frame(maxWidth: 520, alignment: .leading)
        .overlay(alignment: .bottomLeading) {
            if model.phase == .ready {
                HStack(spacing: 6) {
                    if model.personalized { Tag(text: "Tuned to you") }
                    if model.contextReady && model.contextEnabled { Tag(text: "Context fix") }
                }
                .offset(y: 28)
            }
        }
    }
}

struct Tag: View {
    let text: String
    var body: some View {
        Text(text)
            .font(.system(size: 11, weight: .semibold))
            .foregroundStyle(.white.opacity(0.75))
            .padding(.horizontal, 8).padding(.vertical, 3)
            .background(.white.opacity(0.12), in: Capsule())
    }
}

struct AccessibilityNudge: View {
    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "keyboard")
            Text("To type into other apps, allow SilentFlow under Accessibility.")
            Button("Open Settings") {
                PushToTalk.requestTrust()
                PushToTalk.openAccessibilitySettings()
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
        }
        .font(.system(size: 13))
        .foregroundStyle(.white.opacity(0.9))
        .padding(.leading, 14).padding(.trailing, 8).padding(.vertical, 6)
        .background(.black.opacity(0.55), in: Capsule())
    }
}

struct CalibrateNudge: View {
    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "sparkles")
            Text("Press **K** to calibrate to your lips. It takes about a minute and makes reading much more accurate.")
        }
        .font(.system(size: 13))
        .foregroundStyle(.white.opacity(0.9))
        .padding(.horizontal, 14).padding(.vertical, 8)
        .background(.black.opacity(0.5), in: Capsule())
    }
}

struct Subtitles: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        VStack(spacing: 6) {
            ForEach(Array(model.captions.suffix(2).enumerated()), id: \.element.id) { idx, cap in
                let isLatest = idx == model.captions.suffix(2).count - 1
                Text(cap.text)
                    .font(.system(size: isLatest ? 32 : 24, weight: .semibold))
                    .foregroundStyle(.white.opacity(cap.final ? (isLatest ? 1 : 0.6) : 0.7))
                    .italic(!cap.final)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 6)
                    .background(.black.opacity(isLatest ? 0.62 : 0.4), in: RoundedRectangle(cornerRadius: 8))
                    .transition(.opacity.combined(with: .move(edge: .bottom)))
            }
            if model.speaking && model.captions.last?.final != false {
                TypingDots().transition(.opacity)
            }
        }
        .animation(.easeOut(duration: 0.2), value: model.captions)
        .animation(.easeOut(duration: 0.2), value: model.speaking)
        .frame(maxWidth: .infinity)
    }
}

struct TypingDots: View {
    @State private var on = false
    var body: some View {
        HStack(spacing: 6) {
            ForEach(0..<3) { i in
                Circle().fill(.white).frame(width: 7, height: 7)
                    .opacity(on ? 0.9 : 0.25)
                    .animation(.easeInOut(duration: 0.5).repeatForever().delay(Double(i) * 0.15), value: on)
            }
        }
        .padding(.horizontal, 14).padding(.vertical, 10)
        .background(.black.opacity(0.5), in: Capsule())
        .onAppear { on = true }
    }
}

struct DebugPanel: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Model input").font(.system(size: 11, weight: .semibold)).foregroundStyle(.white.opacity(0.7))
            Group {
                if let img = model.mouthImage {
                    Image(decorative: img, scale: 1)
                        .resizable()
                        .interpolation(.medium)
                        .scaleEffect(x: -1, y: 1)
                } else {
                    Color.white.opacity(0.08)
                }
            }
            .frame(width: 144, height: 144)
            .clipShape(RoundedRectangle(cornerRadius: 6))

            Text("Lip motion").font(.system(size: 11, weight: .semibold)).foregroundStyle(.white.opacity(0.7))
            GeometryReader { geo in
                let scale = 0.3
                ZStack(alignment: .leading) {
                    Capsule().fill(.white.opacity(0.12))
                    Capsule().fill(model.motion > model.threshold ? Color.red : Color.green)
                        .frame(width: geo.size.width * min(model.motion / scale, 1))
                    Rectangle().fill(.white)
                        .frame(width: 2)
                        .offset(x: geo.size.width * min(model.threshold / scale, 1))
                }
            }
            .frame(width: 144, height: 6)

            Group {
                Text(String(format: "threshold %.2f  ([ / ])", model.threshold))
                if let ms = model.lastLatencyMs { Text(String(format: "inference %.0f ms on %@", ms, model.device)) }
            }
            .font(.system(size: 11, design: .monospaced))
            .foregroundStyle(.white.opacity(0.65))
        }
        .padding(12)
        .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 10))
    }
}
