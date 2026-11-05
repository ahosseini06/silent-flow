import Foundation

/// Port of Auto-AVSR's mouth ROI preprocessing:
/// smooth landmarks over a short window, fit a similarity transform from the
/// 4 stable points (eyes, nose, mouth) onto the mean reference face (256x256),
/// then cut a 96x96 grayscale patch centred on the mouth.
final class MouthCropper {
    static let side = 96

    /// Stable points of `20words_mean_face.npy` (right eye, left eye, nose, mouth).
    private static let reference: [Point2] = [
        Point2(102.07394306, 94.27230352),
        Point2(156.36130542, 93.57815605),
        Point2(129.00373787, 135.90343029),
        Point2(129.31337323, 157.82299635),
    ]

    /// Auto-AVSR uses a centred ±6 frame window; we use a causal one so there is
    /// no extra latency. Translation still follows the current frame exactly.
    private let window = 7

    /// Calibration knobs: zoom > 1 crops tighter around the mouth; shiftY moves
    /// the crop down (in reference-face pixels). 1 / 0 reproduce Auto-AVSR.
    var zoom = Double(ProcessInfo.processInfo.environment["LIPREAD_CROP_ZOOM"] ?? "") ?? 1.0
    var shiftY = Double(ProcessInfo.processInfo.environment["LIPREAD_CROP_SHIFT_Y"] ?? "") ?? 0.0
    private var history: [[Point2]] = []

    func reset() { history.removeAll() }

    func crop(luma: UnsafePointer<UInt8>, width: Int, height: Int, bytesPerRow: Int,
              points: [Point2]) -> [UInt8] {
        history.append(points)
        if history.count > window { history.removeFirst(history.count - window) }

        var smoothed = [Point2](repeating: .zero, count: 4)
        for frame in history { for i in 0..<4 { smoothed[i] += frame[i] } }
        for i in 0..<4 { smoothed[i] /= Double(history.count) }
        let shift = centroid(points) - centroid(smoothed)
        for i in 0..<4 { smoothed[i] += shift }

        // Similarity transform q = [a -b; b a] p + t  (least squares, Umeyama w/o reflection)
        let ms = centroid(smoothed), md = centroid(Self.reference)
        var num1 = 0.0, num2 = 0.0, den = 0.0
        for i in 0..<4 {
            let s = smoothed[i] - ms, d = Self.reference[i] - md
            num1 += s.x * d.x + s.y * d.y
            num2 += s.x * d.y - s.y * d.x
            den += s.x * s.x + s.y * s.y
        }
        let a = den > 0 ? num1 / den : 1, b = den > 0 ? num2 / den : 0
        let tx = md.x - (a * ms.x - b * ms.y)
        let ty = md.y - (b * ms.x + a * ms.y)

        // Mouth centre in the warped (256x256) face frame
        let m = smoothed[3]
        let cx = a * m.x - b * m.y + tx
        let cy = b * m.x + a * m.y + ty
        let half = Double(Self.side / 2)
        let x0 = (cx - half).rounded().clamped(0, 256)
        let y0 = (cy + shiftY - half).rounded().clamped(0, 256)

        // Inverse map: p = (1/(a²+b²)) [a b; -b a] (q - t)
        let k = 1.0 / (a * a + b * b)
        let ia = a * k, ib = b * k
        let maxX = Double(width - 1), maxY = Double(height - 1)

        var out = [UInt8](repeating: 0, count: Self.side * Self.side)
        out.withUnsafeMutableBufferPointer { dst in
            for v in 0..<Self.side {
                let qy = y0 + half + (Double(v) - half) / zoom
                if qy >= 256 { continue }
                for u in 0..<Self.side {
                    let qx = x0 + half + (Double(u) - half) / zoom
                    if qx >= 256 { continue }
                    let dx = qx - tx, dy = qy - ty
                    let px = ia * dx + ib * dy
                    let py = -ib * dx + ia * dy
                    if px < 0 || py < 0 || px > maxX || py > maxY { continue }
                    let ix = Int(px), iy = Int(py)
                    let fx = px - Double(ix), fy = py - Double(iy)
                    let ix1 = min(ix + 1, width - 1), iy1 = min(iy + 1, height - 1)
                    let r0 = luma + iy * bytesPerRow, r1 = luma + iy1 * bytesPerRow
                    let top = Double(r0[ix]) * (1 - fx) + Double(r0[ix1]) * fx
                    let bot = Double(r1[ix]) * (1 - fx) + Double(r1[ix1]) * fx
                    dst[v * Self.side + u] = UInt8((top * (1 - fy) + bot * fy).rounded().clamped(0, 255))
                }
            }
        }
        return out
    }

    private func centroid(_ p: [Point2]) -> Point2 { p.reduce(.zero, +) / Double(p.count) }
}

extension Double {
    func clamped(_ lo: Double, _ hi: Double) -> Double { Swift.min(Swift.max(self, lo), hi) }
}
