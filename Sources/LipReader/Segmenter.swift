import Foundation

/// Splits the continuous stream of mouth crops into utterances using lip motion
/// (a visual "voice activity detector"), and decides when to ask the model for
/// a live partial transcript vs. the final one.
final class Segmenter {
    enum Event {
        case started(segment: Int)
        case request(segment: Int, final: Bool, frames: Data, count: Int, start: Double)
        case discarded(segment: Int)
    }

    /// Min change in lip opening (fraction of mouth width) over `motionWindow`
    /// to count as speaking.
    var threshold = 0.06
    /// Push-to-talk: while true, everything is treated as speech.
    var manual = false

    let fps = 25.0
    private let motionWindow = 0.45
    private let preRoll = 0.5
    private let hangover = 0.5
    private let minActive = 0.25
    private let maxLength = 10.0
    private let partialEvery = 0.6
    private let keepSeconds = 15.0

    private struct Crop { let t: Double; let pixels: [UInt8]; let openness: Double }
    private struct Segment {
        let id: Int
        var start: Double
        var firstActive: Double
        var lastActive: Double
        var lastPartial: Double
    }

    private var crops: [Crop] = []
    private var opens: [(t: Double, v: Double)] = []
    private var segment: Segment?
    private var nextId = 1
    private var lastFaceTime = -Double.infinity
    private(set) var motion = 0.0

    var isSpeaking: Bool { segment != nil }

    func push(t: Double, crop: [UInt8], openness: Double) -> [Event] {
        lastFaceTime = t
        crops.append(Crop(t: t, pixels: crop, openness: openness))
        let oldestNeeded = min(t - keepSeconds, (segment?.start ?? t) - 0.1)
        if let first = crops.firstIndex(where: { $0.t >= oldestNeeded }), first > 0 {
            crops.removeFirst(first)
        }

        opens.append((t, openness))
        opens.removeAll { $0.t < t - motionWindow }
        let values = opens.map(\.v)
        motion = (values.max() ?? 0) - (values.min() ?? 0)
        let active = manual || motion > threshold

        var events: [Event] = []
        if active {
            if segment == nil {
                segment = Segment(id: nextId, start: t - preRoll, firstActive: t, lastActive: t, lastPartial: t)
                nextId += 1
                events.append(.started(segment: segment!.id))
            } else {
                segment!.lastActive = t
            }
        }

        guard var s = segment else { return events }
        if !manual && t - s.lastActive > hangover {
            events += finish(end: s.lastActive)  // motion window already lags speech end
        } else if t - s.start > maxLength {
            // Too long: cut at the most-closed mouth in the last couple of seconds
            // (most likely a pause between words) and carry on with a new segment.
            let split = quietestMoment(from: t - 2.5, to: t - 0.2) ?? t
            events += finish(end: split)
            segment = Segment(id: nextId, start: split, firstActive: split, lastActive: t, lastPartial: t)
            nextId += 1
            events.append(.started(segment: segment!.id))
        } else if t - s.lastPartial > partialEvery && t - s.start > 1.0 {
            s.lastPartial = t
            segment = s
            if let (data, n) = clip(from: s.start, to: t) {
                events.append(.request(segment: s.id, final: false, frames: data, count: n, start: s.start))
            }
        }
        return events
    }

    func noFace(t: Double) -> [Event] {
        opens.removeAll()
        motion = 0
        if segment != nil && t - lastFaceTime > 1.0 {
            return finish(end: lastFaceTime)
        }
        return []
    }

    /// Push-to-talk pressed: drop any auto-detected segment and start a fresh one,
    /// so what gets typed is exactly what was mouthed while the key was held.
    func beginManual(t: Double) -> [Event] {
        var events: [Event] = []
        if let s = segment {
            segment = nil
            events.append(.discarded(segment: s.id))
        }
        manual = true
        segment = Segment(id: nextId, start: t - 0.3, firstActive: t, lastActive: t, lastPartial: t)
        nextId += 1
        events.append(.started(segment: segment!.id))
        return events
    }

    /// Push-to-talk aborted (the key turned out to be part of a shortcut).
    func cancelManual() -> [Event] {
        manual = false
        guard let s = segment else { return [] }
        segment = nil
        return [.discarded(segment: s.id)]
    }

    /// Push-to-talk released.
    func endManual(t: Double) -> [Event] {
        manual = false
        guard segment != nil else { return [] }
        return finish(end: min(t, lastFaceTime))
    }

    private func finish(end: Double) -> [Event] {
        guard let s = segment else { return [] }
        segment = nil
        if s.lastActive - s.firstActive < minActive {
            return [.discarded(segment: s.id)]
        }
        if let (data, n) = clip(from: s.start, to: end), n >= 12 {
            return [.request(segment: s.id, final: true, frames: data, count: n, start: s.start)]
        }
        return [.discarded(segment: s.id)]
    }

    private func quietestMoment(from a: Double, to b: Double) -> Double? {
        crops.filter { $0.t >= a && $0.t <= b }.min { $0.openness < $1.openness }?.t
    }

    /// Resample crops onto a 25 fps grid (nearest neighbour), as the model expects.
    private func clip(from start: Double, to end: Double) -> (Data, Int)? {
        guard !crops.isEmpty, end > start else { return nil }
        let frameBytes = MouthCropper.side * MouthCropper.side
        let n = Int(((end - start) * fps).rounded(.down))
        guard n > 0 else { return nil }
        var data = Data(capacity: n * frameBytes)
        var j = 0
        for i in 0..<n {
            let tt = start + Double(i) / fps
            while j + 1 < crops.count && abs(crops[j + 1].t - tt) <= abs(crops[j].t - tt) { j += 1 }
            data.append(contentsOf: crops[j].pixels)
        }
        return (data, n)
    }
}
