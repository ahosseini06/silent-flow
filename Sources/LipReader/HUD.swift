import AppKit
import Combine
import SwiftUI

/// A panel that can take keyboard focus without activating the app, so the app
/// you're dictating into stays in front (like Spotlight).
final class DictationPanel: NSPanel {
    override var canBecomeKey: Bool { true }
}

/// The dictation popup at the bottom of the screen: listening state and live
/// reading while the key is held, then an editable box (Return inserts, Esc discards).
final class HUDController {
    private let panel: DictationPanel
    private var sub: AnyCancellable?
    private let size = NSSize(width: 760, height: 220)

    init(model: AppModel) {
        panel = DictationPanel(contentRect: NSRect(origin: .zero, size: size),
                               styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: true)
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = false
        panel.level = .statusBar
        panel.hidesOnDeactivate = false
        panel.isMovableByWindowBackground = false
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary, .ignoresCycle]
        panel.contentView = NSHostingView(rootView: HUDView().environmentObject(model))
        sub = model.$dictation.receive(on: RunLoop.main).sink { [weak self] d in self?.update(d) }
    }

    private func update(_ d: Dictation?) {
        guard let d else { panel.orderOut(nil); return }
        if !panel.isVisible {
            let mouse = NSEvent.mouseLocation
            let screen = NSScreen.screens.first { NSMouseInRect(mouse, $0.frame, false) } ?? NSScreen.main
            if let f = screen?.visibleFrame {
                panel.setFrameOrigin(NSPoint(x: f.midX - size.width / 2, y: f.minY + 30))
            }
        }
        let editing = d.stage == .editing
        panel.ignoresMouseEvents = !editing
        if editing {
            if !panel.isKeyWindow { panel.makeKeyAndOrderFront(nil) }
        } else {
            panel.orderFrontRegardless()
        }
    }
}

struct HUDView: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        VStack {
            Spacer(minLength: 0)
            if let d = model.dictation {
                card(d).transition(.opacity.combined(with: .scale(scale: 0.97)))
            }
        }
        .frame(width: 760, height: 220)
        .animation(.easeOut(duration: 0.15), value: model.dictation?.stage)
    }

    @ViewBuilder
    private func card(_ d: Dictation) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            switch d.stage {
            case .listening:
                row {
                    PulsingDot()
                    Text(d.display.isEmpty
                         ? (model.faceVisible ? "Listening. Mouth your words." : "I can't see your face")
                         : d.display)
                        .italic(!d.live.isEmpty)
                }
            case .reading:
                row {
                    ProgressView().controlSize(.small).tint(.white)
                    Text(d.display.isEmpty ? "Reading…" : d.display).italic()
                }
            case .editing:
                EditBox(text: d.text)
                HStack(spacing: 14) {
                    Hint(key: "↩", label: "insert")
                    Hint(key: "esc", label: "discard")
                    Hint(key: model.pttKey.short, label: "hold to add more")
                    Spacer()
                }
            case .nothing(let msg):
                row {
                    Image(systemName: "exclamationmark.circle.fill").foregroundStyle(.orange)
                    Text(msg)
                }
            }
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 12)
        .frame(maxWidth: 720, alignment: .leading)
        .fixedSize(horizontal: d.stage != .editing, vertical: true)
        .background(.black.opacity(0.84), in: RoundedRectangle(cornerRadius: 16))
        .overlay(RoundedRectangle(cornerRadius: 16).strokeBorder(.white.opacity(0.14)))
        .padding(.bottom, 8)
    }

    private func row<C: View>(@ViewBuilder _ content: () -> C) -> some View {
        HStack(spacing: 10) { content() }
            .font(.system(size: 17, weight: .medium))
            .foregroundStyle(.white)
            .lineLimit(3)
    }
}

/// The editable text. Return inserts, Esc discards.
struct EditBox: View {
    @EnvironmentObject var model: AppModel
    let text: String
    @FocusState private var focused: Bool

    var body: some View {
        TextField("", text: Binding(get: { model.dictation?.text ?? text },
                                    set: { model.editDictation($0) }),
                  axis: .vertical)
            .textFieldStyle(.plain)
            .font(.system(size: 19, weight: .medium))
            .foregroundStyle(.white)
            .lineLimit(1...5)
            .focused($focused)
            .onSubmit { model.commitDictation() }
            .onExitCommand { model.discardDictation() }
            .onAppear { DispatchQueue.main.async { focused = true } }
    }
}

struct Hint: View {
    let key: String
    let label: String
    var body: some View {
        HStack(spacing: 5) {
            Text(key)
                .font(.system(size: 11, weight: .semibold, design: .rounded))
                .padding(.horizontal, 6).padding(.vertical, 2)
                .background(.white.opacity(0.14), in: RoundedRectangle(cornerRadius: 4))
            Text(label).font(.system(size: 12))
        }
        .foregroundStyle(.white.opacity(0.6))
    }
}

struct PulsingDot: View {
    @State private var on = false
    var body: some View {
        Circle().fill(.red).frame(width: 10, height: 10)
            .opacity(on ? 1 : 0.4)
            .animation(.easeInOut(duration: 0.5).repeatForever(), value: on)
            .onAppear { on = true }
    }
}

/// Menu bar item: the app keeps working as a "keyboard" with the window closed.
struct MenuBarContent: View {
    @EnvironmentObject var model: AppModel
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        Text(statusLine)
        if !model.accessibilityTrusted {
            Button("Allow Accessibility Access…") {
                PushToTalk.requestTrust()
                PushToTalk.openAccessibilitySettings()
            }
        }
        Divider()
        Picker("Push-to-talk key", selection: $model.pttKey) {
            ForEach(PTTKey.allCases) { Text($0.label).tag($0) }
        }
        Toggle("Type into the focused app", isOn: $model.typingEnabled)
        Toggle("Context correction", isOn: Binding(get: { model.contextEnabled },
                                                   set: { model.setContextCorrection($0) }))
        Divider()
        Button("Show Window") { showWindow() }
        Button("Calibrate to My Lips…") {
            showWindow()
            model.startCalibration()
        }
        Button("Show Welcome Tour…") {
            showWindow()
            model.replayOnboarding()
        }
        Divider()
        Button("Quit Lip Reader") { NSApp.terminate(nil) }.keyboardShortcut("q")
    }

    private var statusLine: String {
        switch model.phase {
        case .starting, .loadingModel: return "Loading…"
        case .error(let msg): return msg
        case .ready:
            if !model.accessibilityTrusted { return "Needs Accessibility access to type" }
            return "Hold \(model.pttKey.short) in any app to type"
        }
    }

    private func showWindow() {
        openWindow(id: "main")
        NSApp.activate(ignoringOtherApps: true)
    }
}
