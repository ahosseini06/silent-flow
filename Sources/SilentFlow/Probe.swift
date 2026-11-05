import AVFoundation
import Foundation

enum Probe {
    static func run(video: URL, outDir: URL) -> Int32 {
        let asset = AVURLAsset(url: video)
        let sem = DispatchSemaphore(value: 0)
        var track: AVAssetTrack?
        Task {
            track = try? await asset.loadTracks(withMediaType: .video).first
            sem.signal()
        }
        sem.wait()
        guard let track, let reader = try? AVAssetReader(asset: asset) else {
            FileHandle.standardError.write("cannot open \(video.path)\n".data(using: .utf8)!)
            return 1
        }
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
        ])
        reader.add(output)
        reader.startReading()

        try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
        let tracker = FaceTracker(), cropper = MouthCropper(), segmenter = Segmenter()
        var crops = Data()
        var log = "t,face,openness,motion,speaking\n"
        var segments: [[String: Any]] = []
        var frames = 0, faces = 0
        let started = Date()

        while let sb = output.copyNextSampleBuffer() {
            guard let pb = CMSampleBufferGetImageBuffer(sb) else { continue }
            let t = CMSampleBufferGetPresentationTimeStamp(sb).seconds
            frames += 1
            var events: [Segmenter.Event] = []
            if let face = tracker.detect(pb) {
                faces += 1
                CVPixelBufferLockBaseAddress(pb, .readOnly)
                let crop = cropper.crop(
                    luma: CVPixelBufferGetBaseAddressOfPlane(pb, 0)!.assumingMemoryBound(to: UInt8.self),
                    width: CVPixelBufferGetWidthOfPlane(pb, 0),
                    height: CVPixelBufferGetHeightOfPlane(pb, 0),
                    bytesPerRow: CVPixelBufferGetBytesPerRowOfPlane(pb, 0),
                    points: face.points)
                CVPixelBufferUnlockBaseAddress(pb, .readOnly)
                crops.append(contentsOf: crop)
                events = segmenter.push(t: t, crop: crop, openness: face.openness)
                log += String(format: "%.3f,1,%.4f,%.4f,%d\n", t, face.openness, segmenter.motion, segmenter.isSpeaking ? 1 : 0)
            } else {
                cropper.reset()
                crops.append(Data(count: MouthCropper.side * MouthCropper.side))
                events = segmenter.noFace(t: t)
                log += String(format: "%.3f,0,0,0,%d\n", t, segmenter.isSpeaking ? 1 : 0)
            }
            for e in events {
                if case let .request(seg, final, data, count, start) = e, final {
                    let name = "segment_\(seg).raw"
                    try? data.write(to: outDir.appendingPathComponent(name))
                    segments.append(["id": seg, "start": start, "end": start + Double(count) / 25.0, "frames": count, "file": name])
                }
            }
        }
        let elapsed = Date().timeIntervalSince(started)
        try? crops.write(to: outDir.appendingPathComponent("crops.raw"))
        try? log.write(to: outDir.appendingPathComponent("frames.csv"), atomically: true, encoding: .utf8)
        if let j = try? JSONSerialization.data(withJSONObject: segments, options: .prettyPrinted) {
            try? j.write(to: outDir.appendingPathComponent("segments.json"))
        }
        print(String(format: "frames=%d faces=%d segments=%d  %.1f ms/frame", frames, faces, segments.count,
                     elapsed * 1000 / Double(max(frames, 1))))
        return 0
    }
}
