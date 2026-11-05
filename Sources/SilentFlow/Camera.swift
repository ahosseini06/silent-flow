import AVFoundation
import AppKit
import SwiftUI

/// Front camera capture. Frames arrive unmirrored (what the model expects);
/// the preview is mirrored in the UI so it feels like a mirror.
final class Camera: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()
    let queue = DispatchQueue(label: "silentflow.camera", qos: .userInteractive)
    private let output = AVCaptureVideoDataOutput()
    var onFrame: ((CMSampleBuffer) -> Void)?

    func start(completion: @escaping (_ error: String?) -> Void) {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            configureAndRun(completion)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                if granted { self.configureAndRun(completion) }
                else { completion("Camera access was denied. Enable it in System Settings → Privacy & Security → Camera.") }
            }
        default:
            completion("Camera access is off. Enable it in System Settings → Privacy & Security → Camera.")
        }
    }

    func stop() {
        queue.async { self.session.stopRunning() }
    }

    private func pickDevice() -> AVCaptureDevice? {
        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera, .external, .continuityCamera],
            mediaType: .video, position: .unspecified)
        let devices = discovery.devices
        return devices.first(where: { $0.deviceType == .builtInWideAngleCamera })
            ?? devices.first
            ?? AVCaptureDevice.default(for: .video)
    }

    private func configureAndRun(_ completion: @escaping (String?) -> Void) {
        queue.async {
            guard let device = self.pickDevice(),
                  let input = try? AVCaptureDeviceInput(device: device) else {
                completion("No camera found.")
                return
            }
            let s = self.session
            s.beginConfiguration()
            if s.canSetSessionPreset(.hd1280x720) { s.sessionPreset = .hd1280x720 } else { s.sessionPreset = .high }
            if s.canAddInput(input) { s.addInput(input) }
            self.output.videoSettings = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
            ]
            self.output.alwaysDiscardsLateVideoFrames = true
            self.output.setSampleBufferDelegate(self, queue: self.queue)
            if s.canAddOutput(self.output) { s.addOutput(self.output) }
            s.commitConfiguration()
            s.startRunning()
            completion(nil)
        }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        onFrame?(sampleBuffer)
    }
}

final class PreviewNSView: NSView {
    let previewLayer: AVCaptureVideoPreviewLayer

    init(session: AVCaptureSession) {
        previewLayer = AVCaptureVideoPreviewLayer(session: session)
        super.init(frame: .zero)
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.backgroundColor = NSColor.black.cgColor
        layer = previewLayer
        wantsLayer = true
    }

    required init?(coder: NSCoder) { fatalError("unused") }
}

struct CameraPreview: NSViewRepresentable {
    let session: AVCaptureSession
    func makeNSView(context: Context) -> PreviewNSView { PreviewNSView(session: session) }
    func updateNSView(_ nsView: PreviewNSView, context: Context) {}
}
