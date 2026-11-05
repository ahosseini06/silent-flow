import Foundation

/// Runs the PyTorch lip-reading sidecar (inference/server.py) and talks to it
/// over stdin/stdout. One job in flight at a time: finals and calibrations are
/// queued, partials are dropped when the model is busy so latency never builds up.
final class InferenceClient {
    enum Event {
        case ready(device: String, personalized: Bool)
        case llmReady
        /// `done`: no further update will follow for this segment (correction finished or skipped)
        case result(segment: Int, text: String, final: Bool, corrected: Bool, done: Bool, millis: Double)
        case progress(Double)
        case calibrated([String: Any])
        case failed(String)
    }

    struct Take {
        let text: String
        let frames: Data
        let count: Int
    }

    private enum Job {
        case decode(segment: Int, final: Bool, frames: Data, count: Int, quiet: Bool)
        case calibrate(takes: [Take], validation: [Int])
    }

    var onEvent: ((Event) -> Void)?
    var partialBeam = 4
    var finalBeam = 10

    private let queue = DispatchQueue(label: "silentflow.inference")
    private var process: Process?
    private var stdin: FileHandle?
    private var ready = false
    private var busy = false
    private var jobs: [Job] = []
    private var lineBuffer = Data()

    static func locateRoot() -> URL? {
        let fm = FileManager.default
        func ok(_ u: URL) -> Bool { fm.fileExists(atPath: u.appendingPathComponent("inference/server.py").path) }
        if let env = ProcessInfo.processInfo.environment["SILENTFLOW_ROOT"], ok(URL(fileURLWithPath: env)) {
            return URL(fileURLWithPath: env)
        }
        if let s = Bundle.main.object(forInfoDictionaryKey: "SilentFlowRoot") as? String, ok(URL(fileURLWithPath: s)) {
            return URL(fileURLWithPath: s)
        }
        var u = Bundle.main.bundleURL
        for _ in 0..<6 {
            u = u.deletingLastPathComponent()
            if ok(u) { return u }
        }
        return nil
    }

    func start() {
        queue.async { self.launch() }
    }

    func stop() {
        queue.sync {
            try? stdin?.close()
            process?.terminate()
            process = nil
        }
    }

    /// `quiet`: a calibration preview; the sidecar skips context correction and conversation context.
    func submit(segment: Int, final: Bool, frames: Data, count: Int, quiet: Bool = false) {
        queue.async {
            if final {
                self.jobs.append(.decode(segment: segment, final: true, frames: frames, count: count, quiet: quiet))
                self.pump()
            } else if self.ready && !self.busy && self.jobs.isEmpty {
                self.run(.decode(segment: segment, final: false, frames: frames, count: count, quiet: quiet))
            }
        }
    }

    func calibrate(takes: [Take], validation: [Int]) {
        queue.async {
            self.jobs.append(.calibrate(takes: takes, validation: validation))
            self.pump()
        }
    }

    /// Fire-and-forget control messages (no reply).
    func resetContext() { control(["type": "reset_context"]) }
    /// The user accepted a dictation: the sidecar stores clip + final text for fine-tuning.
    func label(segments: [Int], modelText: String, finalText: String) {
        control(["type": "label", "ids": segments, "model_text": modelText, "final_text": finalText])
    }
    func setContextCorrection(_ on: Bool) { control(["type": "config", "llm": on]) }

    // MARK: - private (on `queue`)

    private func control(_ obj: [String: Any]) {
        queue.async {
            guard self.process != nil, var d = try? JSONSerialization.data(withJSONObject: obj) else { return }
            d.append(0x0A)
            self.write(d)
        }
    }

    private func launch() {
        guard let root = Self.locateRoot() else {
            onEvent?(.failed("Can't find the inference/ folder. Set SILENTFLOW_ROOT to the repo path."))
            return
        }
        let python = root.appendingPathComponent("inference/.venv/bin/python")
        guard FileManager.default.isExecutableFile(atPath: python.path) else {
            onEvent?(.failed("Python env missing. Run ./scripts/setup.sh in \(root.path)"))
            return
        }
        let p = Process()
        p.executableURL = python
        p.arguments = [root.appendingPathComponent("inference/server.py").path,
                       "--models", root.appendingPathComponent("models").path]
        p.currentDirectoryURL = root.appendingPathComponent("inference")
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        env["HF_HUB_OFFLINE"] = "1"  // everything is downloaded by setup.sh; never phone home
        p.environment = env

        let inPipe = Pipe(), outPipe = Pipe(), errPipe = Pipe()
        p.standardInput = inPipe
        p.standardOutput = outPipe
        p.standardError = errPipe

        outPipe.fileHandleForReading.readabilityHandler = { [weak self] h in
            let d = h.availableData
            guard !d.isEmpty else { h.readabilityHandler = nil; return }
            self?.queue.async { self?.consume(d) }
        }
        errPipe.fileHandleForReading.readabilityHandler = { h in
            let d = h.availableData
            if d.isEmpty { h.readabilityHandler = nil } else { FileHandle.standardError.write(d) }
        }
        p.terminationHandler = { [weak self] proc in
            self?.queue.async {
                guard let self, self.process === proc else { return }
                self.ready = false
                self.onEvent?(.failed("Lip-reading engine stopped (exit \(proc.terminationStatus))."))
            }
        }
        do {
            try p.run()
        } catch {
            onEvent?(.failed("Couldn't start the engine: \(error.localizedDescription)"))
            return
        }
        process = p
        stdin = inPipe.fileHandleForWriting
    }

    private func consume(_ data: Data) {
        lineBuffer.append(data)
        while let nl = lineBuffer.firstIndex(of: 0x0A) {
            let line = lineBuffer[lineBuffer.startIndex..<nl]
            lineBuffer.removeSubrange(lineBuffer.startIndex...nl)
            guard let obj = try? JSONSerialization.jsonObject(with: line) as? [String: Any],
                  let type = obj["type"] as? String else { continue }
            switch type {
            case "ready":
                ready = true
                onEvent?(.ready(device: obj["device"] as? String ?? "?",
                                personalized: obj["profile"] as? Bool ?? false))
                pump()
            case "llm_ready":
                onEvent?(.llmReady)
            case "result":
                let done = obj["done"] as? Bool ?? true
                if done { busy = false }
                let ms = ((obj["encode_ms"] as? Double) ?? 0) + ((obj["search_ms"] as? Double) ?? 0)
                    + ((obj["llm_ms"] as? Double) ?? 0)
                onEvent?(.result(segment: obj["id"] as? Int ?? -1,
                                 text: obj["text"] as? String ?? "",
                                 final: obj["final"] as? Bool ?? true,
                                 corrected: obj["corrected"] as? Bool ?? false,
                                 done: done,
                                 millis: ms))
                pump()
            case "progress":
                let step = obj["step"] as? Double ?? 0, total = obj["total"] as? Double ?? 1
                onEvent?(.progress(step / max(total, 1)))
            case "calibrated":
                busy = false
                onEvent?(.calibrated(obj))
                pump()
            default:
                break
            }
        }
    }

    private func pump() {
        guard ready, !busy, !jobs.isEmpty else { return }
        run(jobs.removeFirst())
    }

    private func run(_ job: Job) {
        var header: [String: Any]
        var body: Data
        switch job {
        case let .decode(segment, final, frames, count, quiet):
            header = ["type": "decode", "id": segment, "frames": count,
                      "beam": final ? finalBeam : partialBeam, "final": final, "quiet": quiet]
            body = frames
        case let .calibrate(takes, validation):
            header = ["type": "calibrate", "val": validation,
                      "phrases": takes.map { ["text": $0.text, "frames": $0.count] }]
            body = Data(capacity: takes.reduce(0) { $0 + $1.frames.count })
            for t in takes { body.append(t.frames) }
        }
        guard var payload = try? JSONSerialization.data(withJSONObject: header) else { return }
        payload.append(0x0A)
        payload.append(body)
        busy = true
        if !write(payload) { busy = false }
    }

    @discardableResult
    private func write(_ data: Data) -> Bool {
        guard let stdin else { return false }
        do {
            try stdin.write(contentsOf: data)
            return true
        } catch {
            onEvent?(.failed("Lost connection to the engine."))
            return false
        }
    }
}
