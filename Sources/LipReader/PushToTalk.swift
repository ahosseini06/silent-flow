import AppKit
import ApplicationServices

/// Which key acts as the system-wide push-to-talk. Modifier keys on their own
/// don't type anything, so holding one is harmless in any app.
enum PTTKey: String, CaseIterable, Identifiable {
    case rightOption, rightCommand, rightControl, rightShift

    var id: String { rawValue }

    var label: String {
        switch self {
        case .rightOption: return "Right Option (⌥)"
        case .rightCommand: return "Right Command (⌘)"
        case .rightControl: return "Right Control (⌃)"
        case .rightShift: return "Right Shift (⇧)"
        }
    }

    var short: String {
        switch self {
        case .rightOption: return "right ⌥"
        case .rightCommand: return "right ⌘"
        case .rightControl: return "right ⌃"
        case .rightShift: return "right ⇧"
        }
    }

    var keyCode: UInt16 {
        switch self {
        case .rightOption: return 61
        case .rightCommand: return 54
        case .rightControl: return 62
        case .rightShift: return 60
        }
    }

    /// Device-dependent modifier bits (NX_DEVICER*KEYMASK) tell left from right.
    var deviceMask: UInt {
        switch self {
        case .rightOption: return 0x40
        case .rightCommand: return 0x10
        case .rightControl: return 0x2000
        case .rightShift: return 0x04
        }
    }
}

/// Watches the push-to-talk key everywhere (global monitor, needs Accessibility
/// access) and inside our own window (local monitor).
final class PushToTalk {
    var key: PTTKey = .rightOption
    /// true on press, false on release
    var onChange: ((Bool) -> Void)?
    /// another key was pressed while holding: it was a shortcut, not dictation
    var onCancel: (() -> Void)?

    private(set) var isDown = false
    private var monitors: [Any] = []
    private(set) var globalInstalled = false

    static var isTrusted: Bool { AXIsProcessTrusted() }

    static func requestTrust() {
        let opts = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary
        _ = AXIsProcessTrustedWithOptions(opts)
    }

    static func openAccessibilitySettings() {
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility") {
            NSWorkspace.shared.open(url)
        }
    }

    /// Safe to call repeatedly; (re)installs the global monitor once trusted.
    func install() {
        if monitors.isEmpty {
            monitors.append(NSEvent.addLocalMonitorForEvents(matching: [.flagsChanged, .keyDown]) { [weak self] e in
                self?.handle(e)
                return e
            } as Any)
        }
        if !globalInstalled && Self.isTrusted {
            if let m = NSEvent.addGlobalMonitorForEvents(matching: [.flagsChanged, .keyDown], handler: { [weak self] e in
                self?.handle(e)
            }) {
                monitors.append(m)
                globalInstalled = true
            }
        }
    }

    private func handle(_ e: NSEvent) {
        switch e.type {
        case .flagsChanged where e.keyCode == key.keyCode:
            let down = e.modifierFlags.rawValue & key.deviceMask != 0
            if down != isDown {
                isDown = down
                onChange?(down)
            }
        case .keyDown where isDown:
            isDown = false
            onCancel?()
        default:
            break
        }
    }
}

/// Types text into whatever app has keyboard focus, as synthetic key events
/// carrying Unicode strings (works in practically every text field; leaves the
/// clipboard alone).
enum Typer {
    static func type(_ text: String) {
        guard !text.isEmpty else { return }
        let source = CGEventSource(stateID: .hidSystemState)
        let units = Array(text.utf16)
        var i = 0
        while i < units.count {
            let chunk = Array(units[i..<min(i + 16, units.count)])
            i += chunk.count
            for down in [true, false] {
                guard let ev = CGEvent(keyboardEventSource: source, virtualKey: 0, keyDown: down) else { continue }
                ev.flags = []  // don't inherit a still-held modifier
                chunk.withUnsafeBufferPointer { ev.keyboardSetUnicodeString(stringLength: chunk.count, unicodeString: $0.baseAddress) }
                ev.post(tap: .cghidEventTap)
            }
            usleep(2000)
        }
    }
}
