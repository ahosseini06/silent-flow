import AVFoundation
import SwiftUI

/// Calibration, in the onboarding's world: no camera feed (unless you ask for it),
/// the thread flowing into the phrase, and each word lighting up as it's read.
struct CalibrationView: View {
    @EnvironmentObject var model: AppModel
    /// Offline snapshots pass a state instead of reading the live model.
    var preview: Calibration?
    @Environment(\.frozenTime) private var frozen
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var start = Date()
    @State private var showCamera = false

    private var cal: Calibration { preview ?? model.calibration ?? Calibration() }

    /// Phrase block: left edge and top, on the shared stage.
    static let left = 214.0
    static let top = OnboardingView.midY - 104      // instruction row
    static let phraseTop = top + 66                  // first line of the phrase / heading
    static let landing = CGPoint(x: left - 16, y: phraseTop + 27)

    var body: some View {
        GeometryReader { geo in
            let W = OnboardingView.canvas.width, H = OnboardingView.canvas.height
            let k = min(geo.size.width / W, geo.size.height / H)
            let ox = (geo.size.width - W * k) / 2, oy = (geo.size.height - H * k) / 2
            TimelineView(.animation(minimumInterval: 1 / 60, paused: frozen != nil)) { tl in
                let t = frozen ?? tl.date.timeIntervalSince(start)
                let thread = ThreadShape.flow(from: CGPoint(x: -ox / k - 30, y: Self.landing.y + 6), to: Self.landing, wave: 9)
                ZStack(alignment: .topLeading) {
                    ZStack {
                        Backdrop()
                        ThreadView(from: thread, to: thread, morph: 1, t: t, energy: energy,
                                   calm: reduceMotion, scale: k, offset: CGPoint(x: ox, y: oy))
                            .frame(width: geo.size.width, height: geo.size.height)
                    }
                    // greyed out until Space is held: the scene comes alive while you mouth
                    .saturation(active ? 1 : 0)
                    .brightness(active ? 0 : -0.035)
                    .animation(.easeOut(duration: 0.45), value: active)

                    ZStack(alignment: .topLeading) {
                        cameraStatus(t)
                        Group {
                            switch cal.stage {
                            case .recording: recording
                            case .training(let p): training(p)
                            case let .done(title, detail, improved): done(title, detail, improved)
                            }
                        }
                        .transition(.stepChange)
                        footer
                    }
                    .frame(width: W, height: H)
                    .scaleEffect(k, anchor: .topLeading)
                    .offset(x: ox, y: oy)
                    .animation(Theme.arrive, value: stageKey)

                    // the webcam, only on request (kept outside the scaled stage: it's a native layer)
                    if showCamera && preview == nil {
                        CameraPreview(session: model.camera.session)
                            .scaleEffect(x: -1, y: 1)
                            .frame(width: 232 * k, height: 130 * k)
                            .clipShape(RoundedRectangle(cornerRadius: 10 * k))
                            .overlay(RoundedRectangle(cornerRadius: 10 * k).strokeBorder(Theme.hairline))
                            .shadow(color: .black.opacity(0.5), radius: 16, y: 8)
                            .position(x: ox + (80 + 116) * k, y: oy + (84 + 65) * k)
                            .transition(.modifier(active: BlurShift(blur: 10, opacity: 0, x: 0),
                                                  identity: BlurShift(blur: 0, opacity: 1, x: 0)))
                    }
                }
            }
        }
        .background(Theme.bg.ignoresSafeArea())
        .preferredColorScheme(.dark)
    }

    private var stageKey: Int {
        switch cal.stage {
        case .recording: return 0
        case .training: return 1
        case .done: return 2
        }
    }

    /// Colour and light are on while Space is held, while showing a reading, and after recording.
    private var active: Bool {
        if case .recording = cal.stage { return model.manualHeld || cal.settled || (preview?.settled ?? false) }
        return true
    }

    private var energy: Double {
        switch cal.stage {
        case .recording: return model.manualHeld ? 1 : (cal.settled ? 0.7 : 0.28)
        case .training: return 1
        case .done: return 0.5
        }
    }

    // MARK: webcam status, top left

    private func cameraStatus(_ t: Double) -> some View {
        let face = preview != nil || model.faceVisible
        let reading = model.manualHeld
        let (label, color): (String, Color) =
            model.cameraStatus != .authorized && preview == nil ? ("Camera is off", Theme.textSoft)
            : !face ? ("Can't find your face", Theme.textSoft)
            : reading ? ("Reading your lips", Theme.signal)
            : ("Ready", Theme.signal.opacity(0.75))
        return HStack(spacing: 9) {
            Circle().fill(color)
                .frame(width: 7, height: 7)
                .shadow(color: Theme.signal.opacity(reading ? 0.9 : 0), radius: 5)
                .opacity(reading ? 0.6 + 0.4 * cos(t * 6) : 1)
            Text(label)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(face ? Theme.text.opacity(0.8) : Theme.textSoft)
                .contentTransition(.opacity)
            Button {
                withAnimation(Theme.quick) { showCamera.toggle() }
            } label: {
                HStack(spacing: 4) {
                    Text(showCamera ? "Hide webcam" : "View webcam")
                    Image(systemName: "chevron.right")
                        .font(.system(size: 9, weight: .semibold))
                        .rotationEffect(.degrees(showCamera ? 90 : 0))
                }
                .font(.system(size: 12))
                .foregroundStyle(Theme.textSoft.opacity(0.6))
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .padding(.leading, 6)
        }
        .animation(Theme.quick, value: label)
        .frame(width: 600, alignment: .leading)
        .position(x: 80 + 300, y: 58)
    }

    // MARK: stages

    private var recording: some View {
        let c = cal
        let n = c.words.count
        let held = model.manualHeld
        let live = active
        let instruction: String = c.settled ? "Got it."
            : held ? "Keep holding while you mouth it. Let go when you're done."
            : !model.faceVisible && preview == nil ? "Sit where the camera can see your face, then hold space."
            : "Hold space, then mouth the phrase below."
        let reading: String = c.settled ? "Read \(c.heard.count) of \(n) words."
            : held ? (c.heard.isEmpty ? "Reading…" : "Reading… \(c.heard.count) of \(n)") : " "
        return VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 16) {
                SpaceKey(pressed: held, done: c.settled)
                Text(instruction)
                    .font(.system(size: 17, weight: .medium))
                    .foregroundStyle(Theme.text)
                    .contentTransition(.opacity)
            }
            .frame(height: 38)
            .animation(Theme.quick, value: instruction)
            .padding(.bottom, 28)

            FlowLayout(alignment: .leading, spacing: 13, lineSpacing: 6) {
                ForEach(Array(c.words.enumerated()), id: \.offset) { i, word in
                    let heard = c.heard.contains(i)
                    Text(word)
                        .font(Theme.display(44))
                        .tracking(-0.8)
                        .foregroundStyle(heard ? Theme.signal : Theme.text.opacity(c.settled ? 0.35 : live ? 0.95 : 0.42))
                        .animation(.easeOut(duration: 0.4), value: live)
                        .shadow(color: Theme.signal.opacity(heard ? 0.5 : 0), radius: 14)
                        .scaleEffect(heard ? 1.03 : 1, anchor: .bottom)
                        .animation(.spring(response: 0.4, dampingFraction: 0.72), value: heard)
                        .animation(.easeOut(duration: 0.35), value: c.settled)
                }
            }
            .id(c.index)
            .transition(.stepChange)
            .accessibilityElement(children: .combine)
            .accessibilityLabel(c.phrase)

            Text(reading)
                .font(.system(size: 15, weight: .medium))
                .foregroundStyle(c.heard.isEmpty ? Theme.textSoft : Theme.signal.opacity(0.95))
                .contentTransition(.numericText())
                .padding(.top, 24)
                .animation(Theme.quick, value: reading)

            if let notice = c.notice {
                Text(notice)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(Theme.signal.opacity(0.85))
                    .padding(.top, 10)
                    .transition(.opacity)
            }
        }
        .animation(Theme.arrive, value: c.index)
        .frame(width: OnboardingView.canvas.width - 2 * Self.left, height: 340, alignment: .topLeading)
        .position(x: OnboardingView.canvas.width / 2, y: Self.top + 170)
    }

    private func training(_ p: Double) -> some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("Tuning to your lips.")
                .font(Theme.display(44))
                .tracking(-0.8)
                .foregroundStyle(Theme.text)
            Text("Adjusting the model to the way you speak. About thirty seconds, and it all stays on this Mac.")
                .font(.system(size: 15))
                .lineSpacing(5)
                .foregroundStyle(Theme.textSoft)
            ZStack(alignment: .leading) {
                Capsule().fill(Theme.text.opacity(0.1))
                Capsule().fill(Theme.signal)
                    .frame(width: max(6, 420 * p))
                    .shadow(color: Theme.signal.opacity(0.7), radius: 6)
                    .animation(.easeOut(duration: 0.6), value: p)
            }
            .frame(width: 420, height: 3)
            .padding(.top, 14)
        }
        .frame(width: OnboardingView.canvas.width - 2 * Self.left, height: 320, alignment: .topLeading)
        .position(x: OnboardingView.canvas.width / 2, y: Self.phraseTop + 160)
    }

    private func done(_ title: String, _ detail: String, _ improved: Bool) -> some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack(spacing: 14) {
                DrawnCheck(done: true, size: 26)
                Text(title + ".")
                    .font(Theme.display(44))
                    .tracking(-0.8)
                    .foregroundStyle(Theme.text)
            }
            Text(detail)
                .font(.system(size: 15))
                .lineSpacing(5)
                .foregroundStyle(Theme.textSoft)
                .fixedSize(horizontal: false, vertical: true)
                .frame(width: 560, alignment: .leading)
        }
        .frame(width: OnboardingView.canvas.width - 2 * Self.left, height: 320, alignment: .topLeading)
        .position(x: OnboardingView.canvas.width / 2, y: Self.phraseTop + 160)
    }

    // MARK: footer

    private var footer: some View {
        let c = cal
        return HStack {
            if case .recording = c.stage {
                HStack(spacing: 6) {
                    ForEach(0..<Calibration.phrases.count, id: \.self) { i in
                        Capsule()
                            .fill(i == c.index ? Theme.signal : Theme.text.opacity(i < c.index ? 0.4 : 0.12))
                            .frame(width: i == c.index ? 24 : 8, height: 3)
                    }
                }
                .animation(Theme.arrive, value: c.index)
                .accessibilityLabel("Phrase \(c.index + 1) of \(Calibration.phrases.count)")
            }
            Spacer()
            switch c.stage {
            case .recording:
                if c.index > 0 || c.settled {
                    Button("Redo last") { model.redoLastPhrase() }.buttonStyle(PillButton(primary: false))
                }
                Button("Cancel") { model.cancelCalibration() }.buttonStyle(PillButton(primary: false))
            case .training:
                EmptyView()
            case .done:
                Button("Continue") { model.dismissCalibration() }
                    .buttonStyle(PillButton())
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding(.horizontal, 80)
        .frame(width: OnboardingView.canvas.width, height: 60)
        .position(x: OnboardingView.canvas.width / 2, y: OnboardingView.canvas.height - 62)
    }
}

/// Lays words out in wrapping lines (each word stays its own view so it can animate).
struct FlowLayout: Layout {
    var alignment: HorizontalAlignment = .center
    var spacing: CGFloat = 8
    var lineSpacing: CGFloat = 4

    private func rows(_ subviews: Subviews, width: CGFloat) -> [[(Int, CGSize)]] {
        var rows: [[(Int, CGSize)]] = [[]]
        var x: CGFloat = 0
        for (i, v) in subviews.enumerated() {
            let size = v.sizeThatFits(.unspecified)
            if x > 0 && x + spacing + size.width > width {
                rows.append([])
                x = 0
            }
            x += (x > 0 ? spacing : 0) + size.width
            rows[rows.count - 1].append((i, size))
        }
        return rows
    }

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let width = proposal.width ?? .infinity
        var w: CGFloat = 0, h: CGFloat = 0
        let rs = rows(subviews, width: width)
        for r in rs {
            var rowW: CGFloat = 0, rowH: CGFloat = 0
            for (_, size) in r { rowW += size.width; rowH = max(rowH, size.height) }
            rowW += spacing * CGFloat(max(r.count - 1, 0))
            w = max(w, rowW)
            h += rowH
        }
        h += lineSpacing * CGFloat(max(rs.count - 1, 0))
        return CGSize(width: min(w, width), height: h)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var y = bounds.minY
        for r in rows(subviews, width: bounds.width) {
            var rowW: CGFloat = spacing * CGFloat(max(r.count - 1, 0)), rowH: CGFloat = 0
            for (_, size) in r { rowW += size.width; rowH = max(rowH, size.height) }
            var x = alignment == .leading ? bounds.minX : bounds.midX - rowW / 2
            for (i, size) in r {
                subviews[i].place(at: CGPoint(x: x, y: y + (rowH - size.height) / 2), proposal: ProposedViewSize(size))
                x += size.width + spacing
            }
            y += rowH + lineSpacing
        }
    }
}

/// The space bar, drawn as a key: glows softly while waiting, presses down while held.
struct SpaceKey: View {
    var pressed: Bool
    var done: Bool
    @State private var breathe = false

    var body: some View {
        Keycap(legend: "space", selected: pressed || done, pressed: pressed, size: CGSize(width: 118, height: 38))
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .strokeBorder(Theme.signal.opacity(pressed || done ? 0 : (breathe ? 0.55 : 0.12)), lineWidth: 1.2)
                    .shadow(color: Theme.signal.opacity(pressed || done ? 0 : (breathe ? 0.35 : 0)), radius: 8)
            )
            .onAppear {
                withAnimation(.easeInOut(duration: 1.3).repeatForever(autoreverses: true)) { breathe = true }
            }
            .accessibilityLabel("Hold the space bar")
    }
}
