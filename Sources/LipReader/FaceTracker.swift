import CoreVideo
import Vision

typealias Point2 = SIMD2<Double>

struct FaceSample {
    /// Stable points in image pixels (top-left origin, unmirrored image), in the
    /// order Auto-AVSR expects: eye on image-left, eye on image-right, nose, mouth center.
    let points: [Point2]
    /// Inner-lip gap divided by mouth width. ~0 when closed.
    let openness: Double
}

/// Face landmarks via Apple Vision (runs on the Neural Engine / GPU).
final class FaceTracker {
    private let request = VNDetectFaceLandmarksRequest()
    private let handler = VNSequenceRequestHandler()

    func detect(_ pixelBuffer: CVPixelBuffer) -> FaceSample? {
        let w = Double(CVPixelBufferGetWidth(pixelBuffer))
        let h = Double(CVPixelBufferGetHeight(pixelBuffer))
        do {
            try handler.perform([request], on: pixelBuffer, orientation: .up)
        } catch {
            return nil
        }
        guard let faces = request.results, !faces.isEmpty else { return nil }
        let face = faces.max { $0.boundingBox.width * $0.boundingBox.height < $1.boundingBox.width * $1.boundingBox.height }!
        guard let lm = face.landmarks,
              let eyeA = lm.leftEye, let eyeB = lm.rightEye,
              let nose = lm.nose, let outer = lm.outerLips, let inner = lm.innerLips else { return nil }

        let size = CGSize(width: w, height: h)
        // Vision's image coordinates have a bottom-left origin; flip to top-left.
        func pts(_ r: VNFaceLandmarkRegion2D) -> [Point2] {
            r.pointsInImage(imageSize: size).map { Point2(Double($0.x), h - Double($0.y)) }
        }
        func mean(_ p: [Point2]) -> Point2 { p.reduce(Point2(0, 0), +) / Double(max(p.count, 1)) }

        let e1 = mean(pts(eyeA)), e2 = mean(pts(eyeB))
        let (eyeLeft, eyeRight) = e1.x < e2.x ? (e1, e2) : (e2, e1)
        let outerPts = pts(outer)
        let innerPts = pts(inner)
        let mouth = mean(outerPts)

        let xs = outerPts.map(\.x)
        let width = (xs.max() ?? 0) - (xs.min() ?? 0)
        let ys = innerPts.map(\.y)
        let gap = (ys.max() ?? 0) - (ys.min() ?? 0)
        let openness = width > 1 ? gap / width : 0

        return FaceSample(points: [eyeLeft, eyeRight, mean(pts(nose)), mouth], openness: openness)
    }
}
