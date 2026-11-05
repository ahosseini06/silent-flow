import AVFoundation
import AppKit
import Combine
import CoreGraphics
import SwiftUI

struct Caption: Identifiable, Equatable {
    let id: Int
    var text: String
    var final: Bool
    var updated: Date
}

/// One-minute calibration: the user mouths known phrases (holding Space), which
/// sets the lip-motion threshold and fine-tunes the model to their lips.
struct Calibration: Equatable {
    enum Stage: Equatable {
        case recording
        case training(progress: Double)
        case done(title: String, detail: String, improved: Bool)
    }

    static let phrases = [
        "I would like a cup of coffee please",
        "What time is the meeting tomorrow",
        "Can you send me the file before lunch",
        "The weather is really nice today",
        "I think we should try a different approach",
        "Thank you very much for your help",
        "Let me know when you are ready to start",
        "My favourite place to visit is the beach",
        "We need to finish this project by Friday",
        "Could you please turn off the lights",
        "I am going for a walk in the park",
        "It was nice to meet you how are you doing",
    ]
    /// Held out to check the personalised model really is better.
    static let validation = [3, 7, 11]

    var stage: Stage = .recording
    var index = 0
    var takes: [InferenceClient.Take] = []
    var notice: String?
    /// The recording in progress (its segment id), the words of the phrase the
    /// model has read so far, and whether we're showing the final reading.
    var segment: Int?
    var heard: Set<Int> = []
    var settled = false

    var phrase: String { Self.phrases[min(index, Self.phrases.count - 1)] }
    var words: [String] { phrase.split(separator: " ").map(String.init) }

    static func == (a: Calibration, b: Calibration) -> Bool {
        a.stage == b.stage && a.index == b.index && a.notice == b.notice && a.takes.count == b.takes.count
            && a.heard == b.heard && a.settled == b.settled
    }

    /// Which words of `target` appear, in order, in `reading` (forgiving of
    /// spelling and one-letter slips: lip reading can't hear "favourite" vs "favorite").
    static func match(target: [String], reading: String) -> Set<Int> {
        func norm(_ s: String) -> String { s.lowercased().filter { $0.isLetter || $0 == "'" } }
        let a = target.map(norm), b = reading.split(separator: " ").map { norm(String($0)) }.filter { !$0.isEmpty }
        func close(_ x: String, _ y: String) -> Bool {
            if x == y { return true }
            let n = min(x.count, y.count)
            guard n >= 4 else { return false }
            return editDistance(x, y) <= (n >= 7 ? 2 : 1)
        }
        // longest common subsequence with fuzzy equality
        var dp = Array(repeating: Array(repeating: 0, count: b.count + 1), count: a.count + 1)
        for i in stride(from: a.count - 1, through: 0, by: -1) {
            for j in stride(from: b.count - 1, through: 0, by: -1) {
                dp[i][j] = close(a[i], b[j]) ? dp[i + 1][j + 1] + 1 : max(dp[i + 1][j], dp[i][j + 1])
            }
        }
        var out = Set<Int>(), i = 0, j = 0
        while i < a.count && j < b.count {
            if close(a[i], b[j]) && dp[i][j] == dp[i + 1][j + 1] + 1 { out.insert(i); i += 1; j += 1 }
            else if dp[i + 1][j] >= dp[i][j + 1] { i += 1 } else { j += 1 }
        }
        return out
    }

    private static func editDistance(_ x: String, _ y: String) -> Int {
        let a = Array(x), b = Array(y)
        var d = Array(0...b.count)
        for i in 1...max(a.count, 1) where !a.isEmpty {
            var prev = d[0]; d[0] = i
            for j in 1...max(b.count, 1) where !b.isEmpty {
                let cur = d[j]
                d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] == b[j - 1] ? 0 : 1))
                prev = cur
            }
        }
        return d[b.count]
    }
}

/// One push-to-talk dictation: hold the key and mouth, let go, check or edit the
/// text in the popup, press Return to type it (Esc throws it away). Holding the
/// key again before Return appends another phrase.
struct Dictation: Equatable {
    enum Stage: Equatable {
        case listening   // key held
        case reading     // key released, waiting for the model
        case editing     // text is in the box; Return inserts it
        case nothing(String)
    }

    let token = UUID()
    var stage: Stage = .listening
    /// Text already accepted into the box before the current recording (append mode).
    var prefix = ""
    /// Live partial reading of the current recording.
    var live = ""
    /// The editable text.
    var text = ""
    var userEdited = false
    /// Segments still being recorded (started, not yet finalized or discarded).
    var open: Set<Int> = []
    /// Finalized segments in order, and the model's text for each.
    var finals: [Int] = []
    var pieces: [Int: String] = [:]
    /// finals[..<prefixUpTo] are already part of `prefix`.
    var prefixUpTo = 0
    var released = false
    /// The app that was in front when dictation started; text goes there.
    var target = NSWorkspace.shared.frontmostApplication

    var modelText: String { finals.compactMap { pieces[$0] }.filter { !$0.isEmpty }.joined(separator: " ") }
    var display: String { [prefix, live].filter { !$0.isEmpty }.joined(separator: " ") }

    mutating func recompute() {
        guard !userEdited else { return }
        let new = finals[prefixUpTo...].compactMap { pieces[$0] }.filter { !$0.isEmpty }.joined(separator: " ")
        text = [prefix, new].filter { !$0.isEmpty }.joined(separator: " ")
    }
}

/// Glues camera → face tracking → mouth crops → segmentation → model → captions.
/// Published state is only touched on the main thread; pipeline state lives on
/// the camera queue.
final class AppModel: ObservableObject {
    static let shared = AppModel()

    enum Phase: Equatable {
        case starting
        /// First launch of a packaged build: fetching the model weights.
        case downloading(done: Int64, total: Int64)
        case loadingModel
        case ready
        case error(String)
    }

    /// "42% of 4.3 GB"
    static func downloadText(_ done: Int64, _ total: Int64) -> String {
        let pct = total > 0 ? Int((Double(done) / Double(total) * 100).rounded(.down)) : 0
        return "\(min(pct, 100))% of \(String(format: "%.1f", Double(total) / 1e9)) GB"
    }

    @Published var phase: Phase = .starting
    @Published var faceVisible = false
    @Published var speaking = false
    @Published var captions: [Caption] = []
    @Published var showDebug = false
    @Published var mouthImage: CGImage?
    @Published var motion = 0.0
    @Published var threshold = 0.06
    @Published var lastLatencyMs: Double?
    @Published var device = ""
    @Published var manualHeld = false
    @Published var personalized = false
    @Published var contextReady = false
    @Published var contextEnabled = true
    @Published var calibration: Calibration?
    @Published var onboarded = UserDefaults.standard.bool(forKey: "onboarded")
    @Published var cameraStatus = AVCaptureDevice.authorizationStatus(for: .video)
    /// Push-to-talk key held during onboarding's "try it" (nothing is recorded).
    @Published var pttPreviewHeld = false
    /// Hands-on tutorial after setup: dictate into a practice field.
    @Published var tutorialActive = false
    @Published var tutorialText = ""
    private var tutorialAfterCalibration = false
    private var cameraStarted = false
    /// Calibration was asked for but the model is still loading: show the calibration
    /// page (not the camera view) until it can start.
    @Published var calibrateWhenReady = false
    @Published var dictation: Dictation?
    @Published var accessibilityTrusted = PushToTalk.isTrusted
    @Published var pttKey: PTTKey = .rightOption {
        didSet { ptt.key = pttKey; defaults.set(pttKey.rawValue, forKey: "pttKey") }
    }
    @Published var typingEnabled = true {
        didSet { defaults.set(typingEnabled, forKey: "typingEnabled") }
    }

    let camera = Camera()
    private let tracker = FaceTracker()
    private let cropper = MouthCropper()
    private let segmenter = Segmenter()
    private let engine = InferenceClient()
    private let ptt = PushToTalk()
    private var hud: HUDController?
    private var trustTimer: AnyCancellable?
    private var keyMonitor: Any?
    private var sweeper: AnyCancellable?
    private let defaults = UserDefaults.standard

    // camera-queue state
    private var frameCount = 0
    private var calibrating = false
    private var restMotion: [Double] = []
    private var speechMotion: [Double] = []
    private var dictating = false

    private init() {}

    func start() {
        signal(SIGPIPE, SIG_IGN)
        installKeyMonitor()

        if let t = defaults.object(forKey: "motionThreshold") as? Double { setThreshold(t, persist: false) }
        contextEnabled = defaults.object(forKey: "contextCorrection") as? Bool ?? true
        typingEnabled = defaults.object(forKey: "typingEnabled") as? Bool ?? true
        pttKey = PTTKey(rawValue: defaults.string(forKey: "pttKey") ?? "") ?? .rightOption

        ptt.onChange = { [weak self] down in self?.hotkey(down) }
        ptt.onCancel = { [weak self] in self?.abortRecording() }
        // During onboarding, permissions are asked for on their own steps, with context.
        if onboarded && !PushToTalk.isTrusted { PushToTalk.requestTrust() }
        ptt.install()
        trustTimer = Timer.publish(every: 2, on: .main, in: .common).autoconnect().sink { [weak self] _ in
            guard let self else { return }
            let trusted = PushToTalk.isTrusted
            if trusted != self.accessibilityTrusted { self.accessibilityTrusted = trusted }
            self.ptt.install()
        }
        hud = HUDController(model: self)

        engine.onEvent = { [weak self] e in self?.handle(e) }
        engine.start()
        if !contextEnabled { engine.setContextCorrection(false) }
        phase = .loadingModel

        camera.onFrame = { [weak self] sb in self?.process(sb) }
        if onboarded || cameraStatus == .authorized { startCamera() }

        // Fade old captions out after a while.
        sweeper = Timer.publish(every: 1, on: .main, in: .common).autoconnect().sink { [weak self] now in
            guard let self else { return }
            let kept = self.captions.filter { !$0.final || now.timeIntervalSince($0.updated) < 9 }
            if kept.count != self.captions.count { self.captions = kept }
        }
    }

    func startCamera() {
        guard !cameraStarted else { return }
        cameraStarted = true
        camera.start { [weak self] error in
            DispatchQueue.main.async {
                guard let self else { return }
                self.cameraStatus = AVCaptureDevice.authorizationStatus(for: .video)
                if let error, self.onboarded { self.phase = .error(error) }
            }
        }
    }

    // MARK: - Onboarding

    func requestCamera() {
        AVCaptureDevice.requestAccess(for: .video) { granted in
            DispatchQueue.main.async {
                self.cameraStatus = AVCaptureDevice.authorizationStatus(for: .video)
                if granted { self.startCamera() }
            }
        }
    }

    func openCameraSettings() {
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera") {
            NSWorkspace.shared.open(url)
        }
    }

    func finishOnboarding(calibrate: Bool) {
        defaults.set(true, forKey: "onboarded")
        startCamera()
        pttPreviewHeld = false
        // Everything in one transaction, so the camera view never flashes in between.
        withAnimation(.easeInOut(duration: 0.9)) {
            onboarded = true
            if calibrate {
                tutorialAfterCalibration = true
                if phase == .ready { startCalibration() } else { calibrateWhenReady = true }
            } else {
                tutorialText = ""
                tutorialActive = true
            }
        }
    }

    func startTutorial() {
        tutorialText = ""
        withAnimation(.easeInOut(duration: 0.9)) { tutorialActive = true }
    }

    func finishTutorial() {
        withAnimation(.easeInOut(duration: 0.9)) { tutorialActive = false }
        tutorialText = ""
    }

    private func calibrationClosed() {
        if tutorialAfterCalibration {  // same transaction as closing calibration: no camera flash
            tutorialAfterCalibration = false
            tutorialText = ""
            tutorialActive = true
        }
    }

    func replayOnboarding() {
        calibration = nil
        withAnimation(.easeInOut(duration: 0.6)) { onboarded = false }
    }

    func shutdown() {
        camera.stop()
        engine.stop()
    }

    // MARK: - Camera queue

    private func process(_ sb: CMSampleBuffer) {
        guard let pb = CMSampleBufferGetImageBuffer(sb) else { return }
        let t = CMSampleBufferGetPresentationTimeStamp(sb).seconds
        frameCount += 1

        var events: [Segmenter.Event]
        var crop: [UInt8]?
        if let face = tracker.detect(pb) {
            CVPixelBufferLockBaseAddress(pb, .readOnly)
            if let base = CVPixelBufferGetBaseAddressOfPlane(pb, 0) {
                crop = cropper.crop(
                    luma: base.assumingMemoryBound(to: UInt8.self),
                    width: CVPixelBufferGetWidthOfPlane(pb, 0),
                    height: CVPixelBufferGetHeightOfPlane(pb, 0),
                    bytesPerRow: CVPixelBufferGetBytesPerRowOfPlane(pb, 0),
                    points: face.points)
            }
            CVPixelBufferUnlockBaseAddress(pb, .readOnly)
            events = crop.map { segmenter.push(t: t, crop: $0, openness: face.openness) } ?? []
            if calibrating {
                if segmenter.manual { speechMotion.append(segmenter.motion) } else { restMotion.append(segmenter.motion) }
            }
        } else {
            cropper.reset()
            events = segmenter.noFace(t: t)
        }
        route(events)

        let hasFace = crop != nil
        let speaking = segmenter.isSpeaking
        let motion = segmenter.motion
        let image = (frameCount % 3 == 0) ? crop.flatMap(Self.grayImage) : nil
        DispatchQueue.main.async {
            if self.faceVisible != hasFace { self.faceVisible = hasFace }
            if self.speaking != speaking { self.speaking = speaking }
            self.motion = motion
            if let image { self.mouthImage = image }
        }
    }

    private func route(_ events: [Segmenter.Event]) {
        for e in events {
            switch e {
            case let .started(segment):
                if dictating { DispatchQueue.main.async { self.dictation?.open.insert(segment) } }
                if calibrating {
                    DispatchQueue.main.async {
                        guard self.calibration?.stage == .recording, self.calibration?.settled == false else { return }
                        self.calibration?.segment = segment
                        self.calibration?.heard = []
                    }
                }
            case let .request(segment, final, frames, count, _):
                if calibrating {
                    // live reading of the phrase while it's being mouthed (the magic), plus the final take
                    engine.submit(segment: segment, final: final, frames: frames, count: count, quiet: true)
                    if final { DispatchQueue.main.async { self.captured(frames: frames, count: count) } }
                } else {
                    if final {
                        DispatchQueue.main.async {
                            if self.dictation?.open.remove(segment) != nil { self.dictation?.finals.append(segment) }
                        }
                    }
                    engine.submit(segment: segment, final: final, frames: frames, count: count)
                }
            case let .discarded(segment):
                DispatchQueue.main.async {
                    if self.calibration != nil {
                        self.calibration?.notice = "That was too short. Hold Space for the whole phrase."
                    }
                    self.captions.removeAll { $0.id == segment && !$0.final }
                    if self.dictation?.open.remove(segment) != nil { self.settleDictation() }
                }
            }
        }
    }

    // MARK: - Engine events

    private func handle(_ e: InferenceClient.Event) {
        DispatchQueue.main.async {
            switch e {
            case let .ready(device, personalized):
                self.device = device
                self.personalized = personalized
                if case .error = self.phase { return }
                self.phase = .ready
                if self.calibrateWhenReady {
                    self.calibrateWhenReady = false
                    self.startCalibration()
                }
            case .llmReady:
                self.contextReady = true
            case let .downloading(done, total):
                if case .error = self.phase { return }
                self.phase = .downloading(done: done, total: total)
            case .loading:
                if case .error = self.phase { return }
                self.phase = .loadingModel
            case let .failed(msg):
                self.phase = .error(msg)
                if case .training = self.calibration?.stage {
                    self.calibration?.stage = .done(title: "Calibration failed", detail: msg, improved: false)
                }
            case let .result(segment, text, final, corrected, done, ms):
                self.lastLatencyMs = ms
                if self.calibrationReading(segment: segment, text: text, final: final) { return }
                self.apply(segment: segment, text: text, final: final, corrected: corrected)
                self.dictationResult(segment: segment, text: text, final: final, done: done)
            case let .progress(p):
                self.calibration?.stage = .training(progress: p)
            case let .calibrated(report):
                self.finishCalibration(report)
            }
        }
    }

    private func apply(segment: Int, text: String, final: Bool, corrected: Bool) {
        if let i = captions.firstIndex(where: { $0.id == segment }) {
            if captions[i].final && !corrected { return }  // a late partial never overwrites a final
            if final && text.isEmpty { captions.remove(at: i); return }
            captions[i].text = text
            captions[i].final = final
            captions[i].updated = Date()
        } else {
            guard !text.isEmpty, !corrected else { return }
            captions.append(Caption(id: segment, text: text, final: final, updated: Date()))
        }
        if captions.count > 3 { captions.removeFirst(captions.count - 3) }
    }

    // MARK: - Calibration

    func startCalibration() {
        guard phase == .ready, calibration == nil else { return }
        captions.removeAll()
        calibration = Calibration()
        camera.queue.async {
            self.calibrating = true
            self.restMotion.removeAll()
            self.speechMotion.removeAll()
            self.segmenter.threshold = .infinity  // only Space starts a recording
        }
    }

    func cancelCalibration() {
        guard let c = calibration else { return }
        if case .training = c.stage { return }  // can't interrupt the training run
        calibration = nil
        let t = threshold
        camera.queue.async {
            self.calibrating = false
            self.segmenter.threshold = t
        }
        calibrationClosed()
    }

    func redoLastPhrase() {
        guard var c = calibration, c.stage == .recording, c.index > 0 || c.settled else { return }
        if c.settled { c.takes.removeLast(); c.settled = false } else { c.index -= 1; c.takes.removeLast() }
        c.notice = nil
        c.heard = []
        c.segment = nil
        calibration = c
    }

    private func captured(frames: Data, count: Int) {
        guard var c = calibration, c.stage == .recording else { return }
        if count < 20 {
            c.notice = "That was too short. Hold Space for the whole phrase."
            calibration = c
            return
        }
        c.takes.append(.init(text: c.phrase, frames: frames, count: count))
        c.notice = nil
        c.settled = true  // keep the phrase up until its final reading lands
        calibration = c
        let index = c.index
        DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) {  // don't wait forever on the reading
            if self.calibration?.index == index, self.calibration?.settled == true { self.advanceCalibration() }
        }
    }

    /// A live or final reading of the phrase being calibrated. Returns true if it was one.
    private func calibrationReading(segment: Int, text: String, final: Bool) -> Bool {
        guard var c = calibration, c.stage == .recording, let seg = c.segment, seg == segment else {
            return calibration != nil  // stale readings during calibration never become captions
        }
        let heard = Calibration.match(target: c.words, reading: text)
        c.heard = final ? heard : c.heard.union(heard)
        calibration = c
        if final && c.settled {
            let index = c.index
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.1) {
                if self.calibration?.index == index, self.calibration?.settled == true { self.advanceCalibration() }
            }
        }
        return true
    }

    private func advanceCalibration() {
        guard var c = calibration, c.stage == .recording else { return }
        c.index += 1
        c.settled = false
        c.heard = []
        c.segment = nil
        if c.index < Calibration.phrases.count {
            calibration = c
            return
        }
        c.stage = .training(progress: 0)
        calibration = c
        let takes = c.takes
        camera.queue.async {
            self.calibrating = false
            let t = Self.threshold(rest: self.restMotion, speech: self.speechMotion)
            DispatchQueue.main.async {
                if let t { self.setThreshold(t) } else { self.setThreshold(self.threshold) }
            }
            self.engine.calibrate(takes: takes, validation: Calibration.validation)
        }
    }

    /// Place the speaking threshold between resting lip jitter and actual mouthing.
    private static func threshold(rest: [Double], speech: [Double]) -> Double? {
        guard rest.count > 30, speech.count > 30 else { return nil }
        func q(_ a: [Double], _ p: Double) -> Double { let s = a.sorted(); return s[Int(Double(s.count - 1) * p)] }
        let restHigh = q(rest, 0.9), speechTypical = q(speech, 0.5)
        guard speechTypical > restHigh else { return nil }
        return min(max(restHigh + 0.3 * (speechTypical - restHigh), 0.02), 0.2)
    }

    private func finishCalibration(_ r: [String: Any]) {
        let kept = r["kept"] as? Bool ?? false
        if let err = r["error"] as? String {
            calibration?.stage = .done(title: "Calibration failed", detail: err, improved: false)
            return
        }
        let before = r["errors_before"] as? Int ?? 0, after = r["errors_after"] as? Int ?? 0
        let words = r["words"] as? Int ?? 0
        if kept {
            personalized = true
            calibration?.stage = .done(
                title: "Tuned to your lips",
                detail: "On three phrases it hadn't trained on, mistakes went from \(before) to \(after) out of \(words) words. "
                    + "It keeps learning from the dictations you accept.",
                improved: true)
        } else {
            calibration?.stage = .done(
                title: "Kept the \(personalized ? "previous" : "standard") model",
                detail: "Tuning didn't beat it on three phrases it hadn't trained on, so nothing changed. "
                    + "Even light on your face helps; you can calibrate again any time from the menu bar.",
                improved: false)
        }
    }

    func dismissCalibration() {
        guard let c = calibration, case .done = c.stage else { return }
        calibration = nil
        calibrationClosed()
    }

    // MARK: - Push-to-talk dictation (system-wide)

    private func hotkey(_ down: Bool) {
        if !onboarded {
            pttPreviewHeld = down
            return
        }
        guard phase == .ready, calibration == nil else { return }
        if down {
            if NSApp.isActive && dictation == nil && !tutorialActive {  // our own window: just show subtitles
                setManual(true)
                return
            }
            if var d = dictation {
                guard d.stage == .editing else { return }  // still reading the last phrase
                d.prefix = d.text  // append to whatever is in the box (including edits)
                d.prefixUpTo = d.finals.count
                d.userEdited = false
                d.live = ""
                d.released = false
                d.stage = .listening
                dictation = d
            } else {
                dictation = Dictation()
            }
            manualHeld = true
            camera.queue.async {
                self.dictating = true
                self.route(self.segmenter.beginManual(t: CMClockGetTime(CMClockGetHostTimeClock()).seconds))
            }
        } else {
            guard dictation?.stage == .listening else {
                if manualHeld { setManual(false) }
                return
            }
            dictation?.released = true
            dictation?.stage = .reading
            manualHeld = false
            camera.queue.async {
                let now = CMClockGetTime(CMClockGetHostTimeClock()).seconds
                self.route(self.segmenter.endManual(t: now))
                self.dictating = false
                DispatchQueue.main.async { self.settleDictation() }
            }
        }
    }

    /// The push-to-talk key turned out to be part of a shortcut.
    private func abortRecording() {
        pttPreviewHeld = false
        guard let d = dictation, d.stage == .listening else { return }
        manualHeld = false
        if d.prefix.isEmpty {
            dictation = nil
        } else {
            dictation?.stage = .editing
            dictation?.live = ""
        }
        camera.queue.async {
            self.dictating = false
            self.route(self.segmenter.cancelManual())
        }
    }

    private func dictationResult(segment: Int, text: String, final: Bool, done: Bool) {
        guard var d = dictation else { return }
        if !final {
            guard d.open.contains(segment) else { return }
            d.live = text
        } else {
            guard d.finals.contains(segment) else { return }
            d.pieces[segment] = text  // raw first, then the context-corrected version
            d.recompute()
        }
        dictation = d
        settleDictation()
    }

    /// Once the key is up and every phrase has a reading, hand the text to the user.
    private func settleDictation() {
        guard var d = dictation, d.released, d.stage == .reading, d.open.isEmpty,
              d.finals.allSatisfy({ d.pieces[$0] != nil }) else { return }
        d.live = ""
        d.recompute()
        if d.text.trimmingCharacters(in: .whitespaces).isEmpty {
            d.stage = .nothing(faceVisible ? "Didn't catch that. Hold the key while you mouth the words."
                                            : "I can't see your face. Look toward the camera.")
            let token = d.token
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.2) {
                if self.dictation?.token == token, case .nothing = self.dictation?.stage { self.dictation = nil }
            }
        } else {
            d.stage = .editing
        }
        dictation = d
    }

    /// The user typed in the box.
    func editDictation(_ text: String) {
        guard dictation?.stage == .editing, dictation?.text != text else { return }
        dictation?.text = text
        dictation?.userEdited = true
    }

    /// Return: type the text into the app that was in front, and keep the pair for training.
    func commitDictation() {
        guard let d = dictation, d.stage == .editing else { return }
        let text = d.text.trimmingCharacters(in: .whitespacesAndNewlines)
        dictation = nil  // closes the popup; focus goes back to the app underneath
        guard !text.isEmpty else { return }
        engine.label(segments: d.finals, modelText: d.modelText, finalText: text)
        if tutorialActive {  // the practice field lives in our own window
            tutorialText = [tutorialText, text].filter { !$0.isEmpty }.joined(separator: " ")
            return
        }
        if typingEnabled {
            if let app = d.target, app != NSRunningApplication.current, !app.isActive { app.activate() }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { Typer.type(text + " ") }
        }
    }

    /// Esc: throw the dictation away.
    func discardDictation() {
        let wasListening = dictation?.stage == .listening
        dictation = nil
        if wasListening {
            manualHeld = false
            camera.queue.async {
                self.dictating = false
                self.route(self.segmenter.cancelManual())
            }
        }
    }

    // MARK: - Keys

    private func installKeyMonitor() {
        keyMonitor = NSEvent.addLocalMonitorForEvents(matching: [.keyDown, .keyUp]) { [weak self] ev in
            guard let self else { return ev }
            if ev.window is DictationPanel { return ev }  // typing in the dictation popup
            if !self.onboarded || self.tutorialActive { return ev }  // onboarding / tutorial own the keyboard
            switch (ev.keyCode, ev.type) {
            case (49, .keyDown):  // space: push-to-talk
                if !ev.isARepeat { self.setManual(true) }
                return nil
            case (49, .keyUp):
                self.setManual(false)
                return nil
            case (53, .keyDown):  // esc
                if case .done = self.calibration?.stage { self.dismissCalibration() } else { self.cancelCalibration() }
                return nil
            case (36, .keyDown):  // return
                self.dismissCalibration()
                return nil
            default:
                break
            }
            guard ev.type == .keyDown, let ch = ev.charactersIgnoringModifiers?.lowercased() else { return ev }
            switch ch {
            case "k": self.startCalibration()
            case "r": self.redoLastPhrase()
            case "d": self.showDebug.toggle()
            case "l": self.toggleContext()
            case "[": self.setThreshold(self.threshold + 0.01)  // less sensitive
            case "]": self.setThreshold(self.threshold - 0.01)  // more sensitive
            case "c":
                self.captions.removeAll()
                self.engine.resetContext()
            default: return ev
            }
            return nil
        }
    }

    private func toggleContext() { setContextCorrection(!contextEnabled) }

    func setContextCorrection(_ on: Bool) {
        contextEnabled = on
        defaults.set(on, forKey: "contextCorrection")
        engine.setContextCorrection(on)
    }

    private func setManual(_ on: Bool) {
        if let c = calibration, c.stage != .recording || (on && c.settled) { return }
        manualHeld = on
        camera.queue.async {
            if on {
                self.route(self.segmenter.beginManual(t: CMClockGetTime(CMClockGetHostTimeClock()).seconds))
            } else {
                let now = CMClockGetTime(CMClockGetHostTimeClock()).seconds
                self.route(self.segmenter.endManual(t: now))
            }
        }
    }

    private func setThreshold(_ v: Double, persist: Bool = true) {
        let clamped = min(max(v, 0.01), 0.3)
        threshold = clamped
        if persist { defaults.set(clamped, forKey: "motionThreshold") }
        camera.queue.async { if !self.calibrating { self.segmenter.threshold = clamped } }
    }

    private static func grayImage(_ pixels: [UInt8]) -> CGImage? {
        let side = MouthCropper.side
        guard let provider = CGDataProvider(data: Data(pixels) as CFData) else { return nil }
        return CGImage(width: side, height: side, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: side,
                       space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGBitmapInfo(rawValue: 0),
                       provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
    }
}
