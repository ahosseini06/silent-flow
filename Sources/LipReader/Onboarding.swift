import AVFoundation
import AppKit
import SwiftUI

// MARK: - The world: one silent thread from you to the screen

enum Theme {
    static let bg = Color(red: 0.055, green: 0.050, blue: 0.045)          // #0E0D0B warm charcoal
    static let raised = Color(red: 0.090, green: 0.082, blue: 0.073)      // #171513
    static let glow = Color(red: 0.180, green: 0.120, blue: 0.075)        // warm light where the thread lands
    static let text = Color(red: 0.953, green: 0.929, blue: 0.894)        // #F3EDE4 warm white
    static let textSoft = Color(red: 0.639, green: 0.604, blue: 0.561)    // #A39A8F (6.9:1 on bg)
    static let hairline = text.opacity(0.10)
    static let signal = Color(red: 1.000, green: 0.737, blue: 0.494)      // #FFBC7E filament amber

    /// Confident arrival: fast start, long soft landing.
    static let arrive = Animation.timingCurve(0.16, 1, 0.3, 1, duration: 0.8)
    static let quick = Animation.timingCurve(0.16, 1, 0.3, 1, duration: 0.35)

    static func display(_ size: CGFloat) -> Font { .system(size: size, weight: .light) }
}

/// Lets offline snapshots freeze every clock in the onboarding at a given second.
private struct FrozenTimeKey: EnvironmentKey { static let defaultValue: Double? = nil }
extension EnvironmentValues {
    var frozenTime: Double? {
        get { self[FrozenTimeKey.self] }
        set { self[FrozenTimeKey.self] = newValue }
    }
}

/// When each character of `text` lands, typed at a steady pace from `start`.
func typeTimes(_ text: String, start: Double, perChar: Double) -> [Double] {
    var t = start
    return text.map { c in
        defer { t += c == " " ? perChar * 0.6 : (c.isPunctuation ? perChar * 2 : perChar) }
        return t
    }
}

// MARK: - The thread

/// Where the thread runs: flowing in from the left edge to a point, or curled into a ring.
enum ThreadShape {
    /// `rise`: approach the landing point from below (keeps the thread clear of text on the left).
    case flow(from: CGPoint, to: CGPoint, wave: Double, rise: Double = 0)
    case ring(center: CGPoint, radius: Double)

    func point(_ s: Double, t: Double) -> CGPoint {
        switch self {
        case let .flow(a, b, wave, rise):
            let c1 = CGPoint(x: a.x + (b.x - a.x) * 0.5, y: a.y)
            let c2 = rise > 0 ? CGPoint(x: b.x - 70, y: b.y + rise) : CGPoint(x: b.x - (b.x - a.x) * 0.38, y: b.y)
            let u = 1 - s
            var p = CGPoint(
                x: u * u * u * a.x + 3 * u * u * s * c1.x + 3 * u * s * s * c2.x + s * s * s * b.x,
                y: u * u * u * a.y + 3 * u * u * s * c1.y + 3 * u * s * s * c2.y + s * s * s * b.y)
            // a slow, calm undulation; pinned at the landing point
            let env = sin(.pi * min(s * 1.15, 1))
            p.y += wave * env * (sin(2 * .pi * (1.6 * s - 0.12 * t)) * 0.7 + sin(2 * .pi * (3.1 * s + 0.07 * t)) * 0.3)
            return p
        case let .ring(c, r):
            let a = -Double.pi / 2 + 2 * Double.pi * s
            let rr = r + 2.2 * sin(6 * a + t * 0.8)
            return CGPoint(x: c.x + cos(a) * rr, y: c.y + sin(a) * rr)
        }
    }

    var landing: CGPoint? {
        if case let .flow(_, b, _, _) = self { return b }
        return nil
    }

    var focus: CGPoint {
        switch self {
        case let .flow(_, b, _, _): return b
        case let .ring(c, _): return c
        }
    }
}

/// The luminous line, signals gliding along it, and a soft bloom where they land.
struct ThreadView: View {
    var from: ThreadShape
    var to: ThreadShape
    var morph: Double          // 0 = `from`, 1 = `to`
    var t: Double
    var energy: Double         // 0…1: how much signal is flowing
    var calm: Bool = false     // reduce motion
    var scale: Double = 1      // canvas -> window transform
    var offset: CGPoint = .zero

    var body: some View {
        Canvas { ctx, size in
            ctx.translateBy(x: offset.x, y: offset.y)
            ctx.scaleBy(x: scale, y: scale)
            let n = 220
            let tt = calm ? 0 : t
            var pts: [CGPoint] = []
            pts.reserveCapacity(n + 1)
            for i in 0...n {
                let s = Double(i) / Double(n)
                let a = from.point(s, t: tt), b = to.point(s, t: tt)
                pts.append(CGPoint(x: a.x + (b.x - a.x) * morph, y: a.y + (b.y - a.y) * morph))
            }
            let fa = from.focus, fb = to.focus
            let focus = CGPoint(x: fa.x + (fb.x - fa.x) * morph, y: fa.y + (fb.y - fa.y) * morph)

            // ambient light where the thread lands
            let glowR = 360.0
            ctx.fill(Path(ellipseIn: CGRect(x: focus.x - glowR, y: focus.y - glowR, width: glowR * 2, height: glowR * 2)),
                     with: .radialGradient(Gradient(colors: [Theme.glow.opacity(0.9), Theme.glow.opacity(0)]),
                                           center: focus, startRadius: 0, endRadius: glowR))

            var line = Path()
            line.addLines(pts)
            var halo = ctx
            halo.addFilter(.blur(radius: 6))
            halo.stroke(line, with: .color(Theme.text.opacity(0.07)), lineWidth: 5)
            ctx.stroke(line, with: .color(Theme.text.opacity(0.22)), style: StrokeStyle(lineWidth: 1, lineCap: .round))

            // signals: soft comets travelling toward the landing point
            let count = 6
            let speed = calm ? 0.06 : 0.16
            var arrival = 0.0
            for k in 0..<count {
                let phase = (t * speed + Double(k) / Double(count)).truncatingRemainder(dividingBy: 1)
                let strength = (k % 2 == 0 ? 1.0 : energy) * (0.35 + 0.65 * energy)
                if strength < 0.02 { continue }
                let tail = 0.07
                let i0 = max(0, Int((phase - tail) * Double(n))), i1 = min(n, Int(phase * Double(n)))
                if i1 > i0 + 1 {
                    for i in i0..<i1 {
                        let f = Double(i - i0) / Double(i1 - i0)  // 0 at tail, 1 at head
                        var seg = Path()
                        seg.move(to: pts[i]); seg.addLine(to: pts[i + 1])
                        ctx.stroke(seg, with: .color(Theme.signal.opacity(strength * f * f * 0.95)),
                                   style: StrokeStyle(lineWidth: 1.4, lineCap: .round))
                    }
                    let hp = pts[i1]
                    var g = ctx
                    g.addFilter(.blur(radius: 5))
                    g.fill(Path(ellipseIn: CGRect(x: hp.x - 5, y: hp.y - 5, width: 10, height: 10)),
                           with: .color(Theme.signal.opacity(0.55 * strength)))
                    ctx.fill(Path(ellipseIn: CGRect(x: hp.x - 1.6, y: hp.y - 1.6, width: 3.2, height: 3.2)),
                             with: .color(Color.white.opacity(0.9 * strength)))
                }
                arrival = max(arrival, strength * exp(-pow((phase - 1) / 0.035, 2)) + strength * exp(-pow(phase / 0.02, 2)) * 0.6)
            }

            // landing bloom (only when the thread lands somewhere)
            let la = from.landing, lb = to.landing
            if la != nil || lb != nil {
                let a = la ?? lb!, b = lb ?? la!
                let w = (la == nil) ? morph : (lb == nil ? 1 - morph : 1)
                let p = CGPoint(x: a.x + (b.x - a.x) * morph, y: a.y + (b.y - a.y) * morph)
                let r = 16 + 10 * arrival
                var bl = ctx
                bl.addFilter(.blur(radius: 8))
                bl.fill(Path(ellipseIn: CGRect(x: p.x - r, y: p.y - r, width: r * 2, height: r * 2)),
                        with: .color(Theme.signal.opacity((0.10 + 0.30 * arrival) * w)))
                ctx.fill(Path(ellipseIn: CGRect(x: p.x - 2.2, y: p.y - 2.2, width: 4.4, height: 4.4)),
                         with: .color(Theme.signal.opacity((0.55 + 0.45 * arrival) * w)))
            }
        }
        .allowsHitTesting(false)
    }
}

// MARK: - Text that condenses out of the signal, glyph by glyph

struct GlyphWrite: TextRenderer {
    var t: Double
    /// When each glyph starts to appear; glyphs past the end are never drawn.
    var times: [Double]
    var exit: Double = 0  // 0…1: whole line dissolves
    var caret: Double = 0
    var caretColor: Color = Theme.signal

    func draw(layout: Text.Layout, in ctx: inout GraphicsContext) {
        var i = 0
        var caretRect: CGRect?
        var firstRect: CGRect?
        for line in layout {
            for run in line {
                for slice in run {
                    let rect = slice.typographicBounds.rect
                    if firstRect == nil { firstRect = rect }
                    let start = i < times.count ? times[i] : .infinity
                    let x = min(max((t - start) / 0.38, 0), 1)
                    if x > 0 {
                        let e = 1 - pow(1 - x, 3)
                        var g = ctx
                        g.opacity = e * (1 - exit)
                        let blur = (1 - e) * 9 + exit * 10
                        if blur > 0.05 { g.addFilter(.blur(radius: blur)) }
                        g.translateBy(x: (1 - e) * -6, y: 0)
                        g.draw(slice)
                        caretRect = rect
                    }
                    i += 1
                }
            }
        }
        if caret > 0.01 {
            let anchor = caretRect.map { CGRect(x: $0.maxX + 3, y: $0.minY, width: 0, height: $0.height) }
                ?? firstRect.map { CGRect(x: $0.minX, y: $0.minY, width: 0, height: $0.height) }
            if let a = anchor {
                let h = a.height * 0.74
                let bar = CGRect(x: a.minX, y: a.minY + (a.height - h) / 2 + a.height * 0.05, width: 1.5, height: h)
                ctx.fill(Path(roundedRect: bar, cornerRadius: 0.75), with: .color(caretColor.opacity(caret * (1 - exit))))
            }
        }
    }
}

// MARK: - Surface

enum Grain {
    static let image: CGImage = {
        let n = 160
        var px = [UInt8](repeating: 0, count: n * n)
        var seed: UInt64 = 0x9E3779B97F4A7C15
        for i in 0..<px.count {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17
            px[i] = UInt8(truncatingIfNeeded: seed >> 56)
        }
        let provider = CGDataProvider(data: Data(px) as CFData)!
        return CGImage(width: n, height: n, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: n,
                       space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGBitmapInfo(rawValue: 0),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    }()
}

struct Backdrop: View {
    var body: some View {
        ZStack {
            Theme.bg
            LinearGradient(colors: [Color(red: 1, green: 0.85, blue: 0.7).opacity(0.025), .clear], startPoint: .top, endPoint: .center)
            Image(decorative: Grain.image, scale: 2)
                .resizable(resizingMode: .tile)
                .opacity(0.03)
                .blendMode(.screen)
        }
        .ignoresSafeArea()
    }
}

// MARK: - Controls

struct PillButton: ButtonStyle {
    var primary = true
    func makeBody(configuration: Configuration) -> some View {
        PillBody(configuration: configuration, primary: primary)
    }

    private struct PillBody: View {
        let configuration: ButtonStyleConfiguration
        let primary: Bool
        @State private var hover = false
        @Environment(\.isEnabled) private var enabled

        var body: some View {
            configuration.label
                .font(.system(size: 14, weight: primary ? .semibold : .medium))
                .foregroundStyle(primary ? Theme.bg : Theme.text.opacity(hover ? 1 : 0.68))
                .padding(.horizontal, primary ? 24 : 12)
                .padding(.vertical, 11)
                .background {
                    if primary {
                        Capsule().fill(Theme.text.opacity(hover ? 1 : 0.92))
                            .shadow(color: Theme.signal.opacity(hover ? 0.22 : 0), radius: 16)
                            .shadow(color: .black.opacity(0.4), radius: 8, y: 4)
                    }
                }
                .scaleEffect(configuration.isPressed ? 0.975 : 1)
                .opacity(enabled ? 1 : 0.4)
                .animation(Theme.quick, value: hover)
                .animation(.easeOut(duration: 0.12), value: configuration.isPressed)
                .onHover { hover = $0 }
                .contentShape(Capsule())
        }
    }
}

struct Keycap: View {
    let legend: String
    var caption: String? = nil
    var selected = false
    var pressed = false
    var size = CGSize(width: 64, height: 52)

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 12)
                .fill(Theme.bg)
                .overlay(RoundedRectangle(cornerRadius: 12)
                    .fill(LinearGradient(colors: [Theme.text.opacity(pressed ? 0.05 : 0.08), Theme.text.opacity(0.025)],
                                         startPoint: .top, endPoint: .bottom)))
                .overlay(RoundedRectangle(cornerRadius: 12)
                    .strokeBorder(selected ? Theme.signal.opacity(pressed ? 0.9 : 0.45) : Theme.text.opacity(0.13),
                                  lineWidth: selected ? 1.2 : 1))
                .shadow(color: Theme.signal.opacity(selected && pressed ? 0.35 : 0), radius: 18)
                .shadow(color: .black.opacity(pressed ? 0.2 : 0.5), radius: pressed ? 1 : 7, y: pressed ? 1 : 5)
            VStack(spacing: 4) {
                Text(legend).font(.system(size: size.height * 0.34, weight: .light))
                if let caption {
                    Text(caption).font(.system(size: 11, weight: .medium)).foregroundStyle(Theme.textSoft)
                }
            }
            .foregroundStyle(Theme.text.opacity(selected ? 1 : 0.8))
        }
        .frame(width: size.width, height: size.height)
        .offset(y: pressed ? 2.5 : 0)
        .animation(.easeOut(duration: 0.09), value: pressed)
    }
}

/// A hairline ring that closes, then a tick that draws itself.
struct DrawnCheck: View {
    var done: Bool
    var size: CGFloat = 22
    var body: some View {
        ZStack {
            Circle().stroke(Theme.text.opacity(0.14), lineWidth: 1.2)
            Circle().trim(from: 0, to: done ? 1 : 0)
                .stroke(Theme.signal.opacity(0.8), style: StrokeStyle(lineWidth: 1.2, lineCap: .round))
                .rotationEffect(.degrees(-90))
                .animation(.easeOut(duration: 0.5), value: done)
            Tick().trim(from: 0, to: done ? 1 : 0)
                .stroke(Theme.text, style: StrokeStyle(lineWidth: 1.6, lineCap: .round, lineJoin: .round))
                .padding(size * 0.3)
                .animation(.easeOut(duration: 0.35).delay(0.4), value: done)
        }
        .frame(width: size, height: size)
    }

    struct Tick: Shape {
        func path(in r: CGRect) -> Path {
            var p = Path()
            p.move(to: CGPoint(x: r.minX, y: r.midY + r.height * 0.05))
            p.addLine(to: CGPoint(x: r.minX + r.width * 0.38, y: r.maxY - r.height * 0.08))
            p.addLine(to: CGPoint(x: r.maxX, y: r.minY + r.height * 0.1))
            return p
        }
    }
}

/// Insertion arrives out of a blur; removal dissolves.
struct BlurShift: ViewModifier {
    var blur: Double
    var opacity: Double
    var x: Double
    func body(content: Content) -> some View {
        content.blur(radius: blur).opacity(opacity).offset(x: x)
    }
}

extension AnyTransition {
    static var stepChange: AnyTransition {
        .asymmetric(
            insertion: .modifier(active: BlurShift(blur: 14, opacity: 0, x: 28), identity: BlurShift(blur: 0, opacity: 1, x: 0)),
            removal: .modifier(active: BlurShift(blur: 10, opacity: 0, x: -20), identity: BlurShift(blur: 0, opacity: 1, x: 0)))
    }
}

// MARK: - Onboarding

struct OnboardingView: View {
    enum Step: Int, CaseIterable { case welcome, camera, access, key, calibrate }

    @EnvironmentObject var model: AppModel
    @Environment(\.frozenTime) private var frozen
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State var step: Step = .welcome
    @State private var previous: Step = .welcome
    @State private var stepStart = Date()
    @State private var origin = Date()
    @State private var appeared = false

    init(step: Step = .welcome) {
        _step = State(initialValue: step)
        _previous = State(initialValue: step)
    }

    static let canvas = CGSize(width: 1080, height: 700)
    /// Centre of the right-hand illustration column.
    static let rightX = 800.0
    static let welcomePhrases = ["Straight from your head to the page.", "No sound. No cloud. Just intent.",
                                 "Mouth it, and it's written."]
    /// Every step's two columns share this centre line.
    static let midY = 330.0
    static let welcomeLeft = 214.0
    static let welcomeOrigin = CGPoint(x: welcomeLeft - 14, y: midY - 62)

    var body: some View {
        GeometryReader { geo in
            // Lay out on a fixed 1080x700 stage, scaled to fit the window and centred in it.
            let W = Self.canvas.width, H = Self.canvas.height
            let k = min(geo.size.width / W, geo.size.height / H)
            let ox = (geo.size.width - W * k) / 2, oy = (geo.size.height - H * k) / 2
            let edge = -ox / k - 30  // the window's left edge, in stage coordinates
            TimelineView(.animation(minimumInterval: 1 / 60, paused: frozen != nil)) { tl in
                let t = frozen ?? tl.date.timeIntervalSince(stepStart)
                let g = frozen ?? tl.date.timeIntervalSince(origin)
                let m = frozen != nil ? 1 : Self.ease(min(max(t / 1.1, 0), 1))
                ZStack(alignment: .topLeading) {
                    Backdrop()

                    ThreadView(from: shape(previous, t: t, edge: edge), to: shape(step, t: t, edge: edge), morph: m,
                               t: g, energy: energy(t), calm: reduceMotion, scale: k, offset: CGPoint(x: ox, y: oy))
                        .frame(width: geo.size.width, height: geo.size.height)

                    ZStack(alignment: .topLeading) {
                        Group {
                            switch step {
                            case .welcome: WelcomeStep(t: t)
                            case .camera: CameraStep(t: t)
                            case .access: AccessStep(t: t)
                            case .key: KeyStep(t: t)
                            case .calibrate: CalibrateStep(t: t)
                            }
                        }
                        .id(step)
                        .transition(.stepChange)

                        footer
                    }
                    .frame(width: W, height: H)
                    .scaleEffect(k, anchor: .topLeading)
                    .offset(x: ox, y: oy)
                }
                .frame(width: geo.size.width, height: geo.size.height)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Theme.bg.ignoresSafeArea())
        .opacity(appeared || frozen != nil ? 1 : 0)
        .onAppear { withAnimation(.easeOut(duration: 1.2)) { appeared = true } }
        .preferredColorScheme(.dark)
    }

    static func ease(_ x: Double) -> Double { 1 - pow(1 - x, 4) }

    // MARK: where the thread goes on each step

    private func shape(_ s: Step, t: Double, edge: Double) -> ThreadShape {
        let low = Self.midY + 230  // threads into the right column rise from below the text
        switch s {
        case .welcome:
            return .flow(from: CGPoint(x: edge, y: Self.welcomeOrigin.y + 8), to: Self.welcomeOrigin, wave: 9)
        case .camera:
            return .ring(center: CGPoint(x: Self.rightX, y: Self.midY), radius: 150)
        case .access:
            return .flow(from: CGPoint(x: edge, y: low), to: CGPoint(x: Self.rightX - 190, y: Self.midY + 23), wave: 7, rise: 140)
        case .key:
            return .flow(from: CGPoint(x: edge, y: low), to: KeyStep.keyAnchor(model.pttKey), wave: 7, rise: 160)
        case .calibrate:
            return .flow(from: CGPoint(x: edge, y: low), to: CGPoint(x: Self.rightX - 214, y: Self.midY), wave: 7, rise: 140)
        }
    }

    private func energy(_ t: Double) -> Double {
        switch step {
        case .welcome:
            let (text, local, _) = Self.welcomeClock(t)
            let end = (typeTimes(text, start: 0.6, perChar: 0.055).last ?? 0) + 0.3
            return local > 0.4 && local < end ? 1 : 0.25
        case .camera: return model.cameraStatus == .authorized ? 0.7 : 0.2
        case .access: return model.accessibilityTrusted ? 0.7 : 0.25
        case .key: return model.pttPreviewHeld ? 1 : 0.2
        case .calibrate: return 0.45
        }
    }

    /// Which welcome phrase is showing and the time inside it.
    static func welcomeClock(_ t: Double) -> (String, Double, Double) {
        let periods = welcomePhrases.map { (typeTimes($0, start: 0.6, perChar: 0.055).last ?? 0) + 3.2 }
        var x = max(t, 0).truncatingRemainder(dividingBy: periods.reduce(0, +))
        for (p, d) in zip(welcomePhrases, periods) {
            if x < d { return (p, x, d) }
            x -= d
        }
        return (welcomePhrases[0], 0, periods[0])
    }

    // MARK: footer

    private var footer: some View {
        HStack(alignment: .center) {
            HStack(spacing: 6) {
                ForEach(Step.allCases, id: \.self) { s in
                    Capsule()
                        .fill(s == step ? Theme.signal : Theme.text.opacity(s.rawValue < step.rawValue ? 0.4 : 0.12))
                        .frame(width: s == step ? 24 : 8, height: 3)
                }
            }
            .animation(Theme.arrive, value: step)
            .accessibilityLabel("Step \(step.rawValue + 1) of \(Step.allCases.count)")

            Spacer()

            if step != .welcome {
                Button("Back") { go(to: Step(rawValue: step.rawValue - 1)!) }
                    .buttonStyle(PillButton(primary: false))
                    .keyboardShortcut(.leftArrow, modifiers: [.command])
            }
            secondaryAction
            primaryAction
        }
        .padding(.horizontal, 80)
        .frame(width: Self.canvas.width, height: 60)
        .position(x: Self.canvas.width / 2, y: Self.canvas.height - 62)
    }

    @ViewBuilder private var secondaryAction: some View {
        switch step {
        case .camera where model.cameraStatus != .authorized:
            Button("Not now") { next() }.buttonStyle(PillButton(primary: false))
        case .access where !model.accessibilityTrusted:
            Button("Skip for now") { next() }.buttonStyle(PillButton(primary: false))
        case .calibrate where !model.personalized:
            Button("Later") { model.finishOnboarding(calibrate: false) }.buttonStyle(PillButton(primary: false))
        case .calibrate:
            Button("Calibrate again") { model.finishOnboarding(calibrate: true) }.buttonStyle(PillButton(primary: false))
        default:
            EmptyView()
        }
    }

    @ViewBuilder private var primaryAction: some View {
        switch step {
        case .welcome:
            Button("Begin") { next() }.buttonStyle(PillButton()).keyboardShortcut(.defaultAction)
        case .camera where model.cameraStatus == .notDetermined:
            Button("Allow camera") { model.requestCamera() }.buttonStyle(PillButton()).keyboardShortcut(.defaultAction)
        case .camera where model.cameraStatus == .denied || model.cameraStatus == .restricted:
            Button("Open Camera settings") { model.openCameraSettings() }.buttonStyle(PillButton())
        case .access where !model.accessibilityTrusted:
            Button("Open Accessibility settings") {
                PushToTalk.requestTrust()
                PushToTalk.openAccessibilitySettings()
            }
            .buttonStyle(PillButton()).keyboardShortcut(.defaultAction)
        case .calibrate:
            if model.personalized {
                Button("Continue") { model.finishOnboarding(calibrate: false) }
                    .buttonStyle(PillButton()).keyboardShortcut(.defaultAction)
            } else {
                Button("Calibrate now") { model.finishOnboarding(calibrate: true) }
                    .buttonStyle(PillButton()).keyboardShortcut(.defaultAction)
            }
        default:
            Button("Continue") { next() }.buttonStyle(PillButton()).keyboardShortcut(.defaultAction)
        }
    }

    private func next() {
        if let n = Step(rawValue: step.rawValue + 1) { go(to: n) }
    }

    private func go(to s: Step) {
        previous = step
        withAnimation(reduceMotion ? .easeOut(duration: 0.25) : Theme.arrive) {
            step = s
            stepStart = Date()
        }
    }
}

// MARK: - Shared step layout

private struct StepText: View {
    let title: String
    let body_: String
    init(_ title: String, _ body: String) { self.title = title; self.body_ = body }

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text(title)
                .font(Theme.display(40))
                .tracking(-0.8)
                .foregroundStyle(Theme.text)
                .fixedSize(horizontal: false, vertical: true)
            Text(body_)
                .font(.system(size: 15))
                .lineSpacing(5)
                .foregroundStyle(Theme.textSoft)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(width: 400, alignment: .leading)
    }
}

private struct LeftColumn<Content: View>: View {
    @ViewBuilder var content: Content
    var body: some View {
        VStack(alignment: .leading, spacing: 30) { content }
            .frame(width: 420, alignment: .leading)
            .position(x: 80 + 210, y: OnboardingView.midY)
    }
}

// MARK: 1. Welcome

private struct WelcomeStep: View {
    let t: Double

    var body: some View {
        let (text, local, period) = OnboardingView.welcomeClock(t)
        let times = typeTimes(text, start: 0.6, perChar: 0.055)
        let exit = min(max((local - (period - 0.8)) / 0.7, 0), 1)
        let writing = local < (times.last ?? 0) + 0.4
        let caret = writing ? 1 : (0.5 + 0.5 * cos(local * 4.2)) * (1 - exit)

        VStack(alignment: .leading, spacing: 0) {
            Text(text)
                .font(Theme.display(46))
                .tracking(-1.1)
                .foregroundStyle(Theme.text)
                .textRenderer(GlyphWrite(t: local, times: times, exit: exit, caret: caret))
                .fixedSize()
                .frame(height: 64, alignment: .leading)
                .accessibilityLabel(OnboardingView.welcomePhrases[0])

            Text("Lip Reader turns the words you mouth into text, in any app on your Mac.\nNo microphone, no cloud. Nothing leaves this machine.")
                .font(.system(size: 16))
                .lineSpacing(6)
                .foregroundStyle(Theme.textSoft)
                .padding(.top, 30)

            Text("Setup takes about a minute.")
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(Theme.textSoft.opacity(0.75))
                .padding(.top, 22)
        }
        .frame(width: 1080 - 2 * OnboardingView.welcomeLeft, alignment: .leading)
        .position(x: 540, y: OnboardingView.midY)
    }
}

// MARK: 3. Camera

private struct CameraStep: View {
    let t: Double
    @EnvironmentObject var model: AppModel

    var body: some View {
        let granted = model.cameraStatus == .authorized
        let denied = model.cameraStatus == .denied || model.cameraStatus == .restricted
        ZStack {
            LeftColumn {
                StepText("It needs to see you.",
                         "Video is read frame by frame on this Mac and never leaves it. Nothing is kept except the dictations you accept, which help it learn how you speak.")
                HStack(spacing: 12) {
                    DrawnCheck(done: granted)
                    Text(granted ? "Camera is on." : denied ? "Camera access is off. Turn it on in System Settings, then come back."
                                                          : "macOS will ask once.")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(granted ? Theme.text : Theme.textSoft)
                }
                .animation(Theme.quick, value: granted)
            }

            // the thread has curled into a lens; it fills with light once allowed
            ZStack {
                Circle()
                    .fill(RadialGradient(colors: [Theme.signal.opacity(granted ? 0.10 : 0.02), .clear],
                                         center: .center, startRadius: 0, endRadius: 140))
                    .animation(.easeInOut(duration: 1.2), value: granted)
                Circle().stroke(Theme.text.opacity(0.05), lineWidth: 1).padding(28)
                Circle().stroke(Theme.text.opacity(0.04), lineWidth: 1).padding(64)
                Circle()
                    .fill(granted ? Theme.signal : Theme.text.opacity(0.25))
                    .frame(width: 7, height: 7)
                    .shadow(color: Theme.signal.opacity(granted ? 0.9 : 0), radius: 8)
                    .animation(.easeInOut(duration: 0.8), value: granted)
            }
            .frame(width: 300, height: 300)
            .position(x: OnboardingView.rightX, y: OnboardingView.midY)
        }
    }
}

// MARK: 4. Accessibility

private struct AccessStep: View {
    let t: Double
    @EnvironmentObject var model: AppModel

    var body: some View {
        let trusted = model.accessibilityTrusted
        // until it's allowed, show the one switch to flip
        let cycle = t.truncatingRemainder(dividingBy: 3.4)
        let demoOn = trusted || (cycle > 1.55 && cycle < 3.0)
        let cursorAtSwitch = cycle >= 1.0

        ZStack {
            LeftColumn {
                StepText("Let it type for you.",
                         "To notice your key in any app and type where your cursor is, macOS needs you to switch Lip Reader on under Privacy & Security, Accessibility.")
                HStack(spacing: 12) {
                    DrawnCheck(done: trusted)
                    Text(trusted ? "Connected. It works in every app." : "Waiting for the switch…")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(trusted ? Theme.text : Theme.textSoft)
                        .opacity(trusted ? 1 : 0.65 + 0.35 * cos(t * 1.8))
                }
                .animation(Theme.quick, value: trusted)
            }

            VStack(alignment: .leading, spacing: 0) {
                Text("Accessibility")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(Theme.textSoft)
                    .padding(.horizontal, 18).padding(.vertical, 14)
                Rectangle().fill(Theme.hairline).frame(height: 1)
                HStack(spacing: 12) {
                    LogoIcon()
                        .frame(width: 30, height: 30)
                    Text("Lip Reader")
                        .font(.system(size: 14, weight: .medium))
                        .foregroundStyle(Theme.text)
                    Spacer()
                    Switch(on: demoOn)
                }
                .padding(.horizontal, 18).padding(.vertical, 14)
                .overlay(alignment: .trailing) {
                    if !trusted {
                        Image(systemName: "cursorarrow")
                            .font(.system(size: 18))
                            .foregroundStyle(Theme.text)
                            .shadow(color: .black.opacity(0.6), radius: 3, y: 1)
                            .offset(x: cursorAtSwitch ? -6 : 44, y: cursorAtSwitch ? 12 : 44)
                            .scaleEffect(cycle > 1.35 && cycle < 1.55 ? 0.85 : 1)
                            .opacity(cycle > 3.0 ? 0 : 1)
                            .animation(.timingCurve(0.16, 1, 0.3, 1, duration: 0.9), value: cursorAtSwitch)
                            .animation(.easeOut(duration: 0.1), value: cycle > 1.35 && cycle < 1.55)
                    }
                }
            }
            .frame(width: 380)
            .background(RoundedRectangle(cornerRadius: 14).fill(Theme.raised))
            .overlay(RoundedRectangle(cornerRadius: 14).strokeBorder(Theme.hairline))
            .shadow(color: .black.opacity(0.45), radius: 24, y: 14)
            .position(x: OnboardingView.rightX, y: OnboardingView.midY)
        }
    }

    private struct Switch: View {
        let on: Bool
        var body: some View {
            Capsule()
                .fill(on ? Theme.signal.opacity(0.85) : Theme.text.opacity(0.14))
                .frame(width: 38, height: 22)
                .overlay(alignment: on ? .trailing : .leading) {
                    Circle().fill(Theme.text).padding(2.5).shadow(color: .black.opacity(0.3), radius: 1, y: 1)
                }
                .animation(.spring(response: 0.3, dampingFraction: 0.82), value: on)
        }
    }
}

// MARK: 5. Push-to-talk key

private struct KeyStep: View {
    let t: Double
    @EnvironmentObject var model: AppModel

    static let gridCenter = CGPoint(x: OnboardingView.rightX, y: OnboardingView.midY)
    static let cap = CGSize(width: 150, height: 96)
    static let gap = 16.0

    static func center(_ k: PTTKey) -> CGPoint {
        let i = PTTKey.allCases.firstIndex(of: k) ?? 0
        let col = Double(i % 2), row = Double(i / 2)
        return CGPoint(x: gridCenter.x + (col - 0.5) * (cap.width + gap),
                       y: gridCenter.y + (row - 0.5) * (cap.height + gap))
    }

    /// Where the thread plugs into a key: its left edge.
    static func keyAnchor(_ k: PTTKey) -> CGPoint {
        let c = center(k)
        return CGPoint(x: c.x - cap.width / 2, y: c.y)
    }

    var body: some View {
        let held = model.pttPreviewHeld
        ZStack {
            LeftColumn {
                StepText("Pick your key.",
                         "A modifier you never press on its own. Hold it while you mouth, let go to read. You can change it later from the menu bar.")
                HStack(spacing: 10) {
                    Circle().fill(held ? Theme.signal : Theme.text.opacity(0.2)).frame(width: 7, height: 7)
                        .shadow(color: Theme.signal.opacity(held ? 0.9 : 0), radius: 6)
                    Text(held ? "Connected. That's the one." : "Try it: hold \(model.pttKey.short) now.")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(held ? Theme.text : Theme.textSoft)
                }
                .animation(Theme.quick, value: held)
            }

            ForEach(PTTKey.allCases) { k in
                Button {
                    withAnimation(Theme.arrive) { model.pttKey = k }
                } label: {
                    Keycap(legend: k.symbol, caption: k.side, selected: model.pttKey == k,
                           pressed: held && model.pttKey == k, size: Self.cap)
                }
                .buttonStyle(.plain)
                .accessibilityLabel(k.label)
                .position(Self.center(k))
            }
        }
    }
}

// MARK: 6. Calibration

private struct CalibrateStep: View {
    let t: Double
    @EnvironmentObject var model: AppModel

    static let rowHeight = 46.0
    static let secondsPerRow = 2.8

    var body: some View {
        ZStack {
            LeftColumn {
                if model.personalized {
                    StepText("Already tuned to you.",
                             "Your calibration lives on this Mac. Run it again any time from the menu bar; each run adds to what it knows.")
                } else {
                    StepText("Teach it how you speak.",
                             "Hold Space and mouth twelve short phrases, and watch each word light up as it reads you. It takes about a minute, and it's the biggest single boost to accuracy.")
                }
            }

            marquee
                .frame(width: 420, height: 400)
                .mask(LinearGradient(stops: [.init(color: .clear, location: 0), .init(color: .black, location: 0.3),
                                             .init(color: .black, location: 0.7), .init(color: .clear, location: 1)],
                                     startPoint: .top, endPoint: .bottom))
                .position(x: OnboardingView.rightX + 10, y: OnboardingView.midY)
        }
    }

    private var marquee: some View {
        let phrases = Calibration.phrases
        let n = phrases.count
        let x = max(t, 0) / Self.secondsPerRow
        let whole = floor(x), frac = x - whole
        let settle = frac < 0.25 ? (1 - pow(1 - frac / 0.25, 3)) : 1
        return ZStack(alignment: .leading) {
            ForEach(-4...4, id: \.self) { k in
                let idx = ((Int(whole) + k) % n + n) % n
                let y = (Double(k) + 1 - settle) * Self.rowHeight
                let focus = max(0, 1 - abs(y) / Self.rowHeight)
                Text(phrases[idx])
                    .font(.system(size: 20, weight: focus > 0.5 ? .regular : .light))
                    .foregroundStyle(Theme.text.opacity(0.2 + 0.8 * focus))
                    .blur(radius: (1 - focus) * 0.6)
                    .offset(x: 6 * (1 - focus), y: y)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .leading)
    }
}

extension PTTKey {
    var symbol: String {
        switch self {
        case .rightOption: return "⌥"
        case .rightCommand: return "⌘"
        case .rightControl: return "⌃"
        case .rightShift: return "⇧"
        }
    }

    var side: String {
        switch self {
        case .rightOption: return "right option"
        case .rightCommand: return "right command"
        case .rightControl: return "right control"
        case .rightShift: return "right shift"
        }
    }
}

extension String {
    var capitalizedFirst: String { prefix(1).uppercased() + dropFirst() }
}

// MARK: - Offline snapshots (`LipReader --render-onboarding <dir>`)

enum OnboardingSnapshots {
    @MainActor
    static func render(to dir: URL) -> Int32 {
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let shots: [(OnboardingView.Step, Double)] = [
            (.welcome, 1.4), (.welcome, 3.4),
            (.camera, 1.0), (.access, 2.0), (.key, 1.0), (.calibrate, 1.4),
        ]
        let sizes: [CGSize] = [OnboardingView.canvas, CGSize(width: 1440, height: 760)]
        for (i, (step, t)) in (shots + [(.welcome, 3.4), (.key, 1.0)]).enumerated() {
            let size = i < shots.count ? sizes[0] : sizes[1]
            let view = OnboardingView(step: step)
                .environmentObject(AppModel.shared)
                .environment(\.frozenTime, t)
                .frame(width: size.width, height: size.height)
            let r = ImageRenderer(content: view)
            r.scale = 2
            guard let img = r.nsImage, let tiff = img.tiffRepresentation,
                  let png = NSBitmapImageRep(data: tiff)?.representation(using: .png, properties: [:]) else { continue }
            let name = "\(step.rawValue)-\(step)-\(String(format: "%.1f", t))-\(Int(size.width)).png"
            try? png.write(to: dir.appendingPathComponent(name))
            print(name)
        }
        // tutorial screen and a calibration card mid-reading
        let extras: [(String, AnyView)] = [
            ("tutorial", AnyView(TutorialView())),
            ("calibration-reading", AnyView({
                var c = Calibration(); c.index = 7; c.heard = [0, 1, 4, 5, 7]
                return CalibrationView(preview: c).environment(\.frozenTime, 2.0)
            }())),
            ("calibration-first", AnyView(CalibrationView(preview: Calibration()).environment(\.frozenTime, 1.0))),
            ("calibration-training", AnyView({
                var c = Calibration(); c.stage = .training(progress: 0.45)
                return CalibrationView(preview: c).environment(\.frozenTime, 2.0)
            }())),
            ("calibration-done", AnyView({
                var c = Calibration()
                c.stage = .done(title: "Tuned to your lips", detail: "Mistakes on your 3 test phrases: 13 → 7 (of 24 words). Lip-motion sensitivity set to 0.09.", improved: true)
                return CalibrationView(preview: c).environment(\.frozenTime, 2.0)
            }())),
        ]
        for (name, v) in extras {
            let r = ImageRenderer(content: v.environmentObject(AppModel.shared)
                .frame(width: OnboardingView.canvas.width, height: OnboardingView.canvas.height))
            r.scale = 2
            if let img = r.nsImage, let tiff = img.tiffRepresentation,
               let png = NSBitmapImageRep(data: tiff)?.representation(using: .png, properties: [:]) {
                try? png.write(to: dir.appendingPathComponent("\(name).png"))
                print(name)
            }
        }
        return 0
    }
}

// MARK: - Hands-on tutorial (after setup and calibration)

/// The real thing, in a safe place: hold the key, mouth a sentence, press Return,
/// and watch it land in a practice message.
struct TutorialView: View {
    @EnvironmentObject var model: AppModel
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var start = Date()
    @State private var landedAt: Date?

    static let prefix = "Running late, "
    static let suggestion = "see you at seven"

    private var stage: Int {
        if !model.tutorialText.isEmpty { return 3 }
        switch model.dictation?.stage {
        case .listening?: return 1
        case .reading?, .editing?: return 2
        default: return 0
        }
    }

    var body: some View {
        GeometryReader { geo in
            let W = OnboardingView.canvas.width, H = OnboardingView.canvas.height
            let k = min(geo.size.width / W, geo.size.height / H)
            let ox = (geo.size.width - W * k) / 2, oy = (geo.size.height - H * k) / 2
            TimelineView(.animation(minimumInterval: 1 / 60)) { tl in
                let t = tl.date.timeIntervalSince(start)
                let field = CGPoint(x: OnboardingView.rightX - 190, y: OnboardingView.midY + 8)
                let energy = stage == 1 ? 1.0 : stage == 3 ? 0.8 : 0.3
                ZStack(alignment: .topLeading) {
                    Backdrop()
                    ThreadView(from: .flow(from: CGPoint(x: -ox / k - 30, y: OnboardingView.midY + 230), to: field, wave: 7, rise: 150),
                               to: .flow(from: CGPoint(x: -ox / k - 30, y: OnboardingView.midY + 230), to: field, wave: 7, rise: 150),
                               morph: 1, t: t, energy: energy, calm: reduceMotion, scale: k, offset: CGPoint(x: ox, y: oy))
                        .frame(width: geo.size.width, height: geo.size.height)

                    ZStack(alignment: .topLeading) {
                        content(t)
                        footer
                    }
                    .frame(width: W, height: H)
                    .scaleEffect(k, anchor: .topLeading)
                    .offset(x: ox, y: oy)
                }
            }
        }
        .background(Theme.bg.ignoresSafeArea())
        .preferredColorScheme(.dark)
        .onChange(of: model.tutorialText) { _, new in if !new.isEmpty { landedAt = Date() } }
    }

    @ViewBuilder
    private func content(_ t: Double) -> some View {
        VStack(alignment: .leading, spacing: 30) {
            VStack(alignment: .leading, spacing: 18) {
                Text(stage == 3 ? "That's all there is to it." : "Now try it for real.")
                    .font(Theme.display(40))
                    .tracking(-0.8)
                    .foregroundStyle(Theme.text)
                    .contentTransition(.opacity)
                Text(stage == 3 ? "It works the same in every app: hold, mouth, Return. Lip Reader lives in your menu bar."
                                : "Dictate the rest of this message, the same way you will in any app. It stays right here.")
                    .font(.system(size: 15))
                    .lineSpacing(5)
                    .foregroundStyle(Theme.textSoft)
                    .fixedSize(horizontal: false, vertical: true)
                    .contentTransition(.opacity)
            }
            VStack(alignment: .leading, spacing: 16) {
                Task(done: stage > 1, active: stage <= 1, text: "Hold \(model.pttKey.short) and mouth \u{201C}\(Self.suggestion)\u{201D}.")
                Task(done: stage > 2, active: stage == 2, text: "Let go. Check the reading, fix it if you like.")
                Task(done: stage > 2, active: false, text: "Press Return to type it.")
            }
            .animation(Theme.quick, value: stage)
        }
        .frame(width: 420, alignment: .leading)
        .position(x: 80 + 210, y: OnboardingView.midY)
        .animation(Theme.arrive, value: stage == 3)

        // the practice message
        let landed = landedAt.map { Date().timeIntervalSince($0) } ?? 0
        let prefixTimes = Array(repeating: -100.0, count: Self.prefix.count)
        let typedTimes = model.tutorialText.indices.enumerated().map { i, _ in 0.05 + Double(i) * 0.03 }
        VStack(alignment: .leading, spacing: 10) {
            Text("Message to Maya")
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(Theme.textSoft)
            Text(Self.prefix + model.tutorialText)
                .font(.system(size: 18))
                .foregroundStyle(Theme.text)
                .textRenderer(GlyphWrite(t: model.tutorialText.isEmpty ? 0 : landed, times: prefixTimes + typedTimes,
                                         caret: 0.5 + 0.5 * cos(t * 5), caretColor: Theme.text))
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 16)
                .padding(.vertical, 15)
                .background(RoundedRectangle(cornerRadius: 12).fill(Theme.raised))
                .overlay(RoundedRectangle(cornerRadius: 12)
                    .strokeBorder(stage == 1 ? Theme.signal.opacity(0.5) : Theme.hairline))
                .animation(Theme.quick, value: stage)
            if stage == 0 && !model.faceVisible {
                Text("I can't see your face yet. Sit in front of the camera.")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(Theme.signal.opacity(0.85))
            }
        }
        .frame(width: 380)
        .position(x: OnboardingView.rightX, y: OnboardingView.midY)
    }

    private var footer: some View {
        HStack {
            Spacer()
            if stage < 3 {
                Button("Skip tutorial") { model.finishTutorial() }.buttonStyle(PillButton(primary: false))
            } else {
                Button("Start using Lip Reader") { model.finishTutorial() }
                    .buttonStyle(PillButton())
                    .keyboardShortcut(.defaultAction)
                    .transition(.opacity)
            }
        }
        .padding(.horizontal, 80)
        .frame(width: OnboardingView.canvas.width, height: 60)
        .position(x: OnboardingView.canvas.width / 2, y: OnboardingView.canvas.height - 62)
    }

    private struct Task: View {
        let done: Bool
        let active: Bool
        let text: String
        var body: some View {
            HStack(spacing: 12) {
                DrawnCheck(done: done, size: 20)
                Text(text)
                    .font(.system(size: 15, weight: active ? .medium : .regular))
                    .foregroundStyle(done ? Theme.textSoft : active ? Theme.text : Theme.textSoft.opacity(0.7))
            }
        }
    }
}
