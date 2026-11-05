import AppKit
import SwiftUI

/// The SilentFlow mark (pixel lips), drawn from the same dots as Resources/logo.svg.
enum LogoMark {
    /// (x, y, opacity) of each 2.25-unit dot, in the logo's 30x30 design space.
    static let dots: [(Double, Double, Double)] = [
        (8.375, 9.675, 0.62),
        (10.975, 9.675, 1.0),
        (13.575, 9.675, 0.32),
        (16.175, 9.675, 0.32),
        (18.775, 9.675, 1.0),
        (21.375, 9.675, 0.62),
        (5.775, 12.275, 0.62),
        (8.375, 12.275, 1.0),
        (10.975, 12.275, 1.0),
        (13.575, 12.275, 1.0),
        (16.175, 12.275, 1.0),
        (18.775, 12.275, 1.0),
        (21.375, 12.275, 1.0),
        (23.975, 12.275, 0.62),
        (3.175, 14.875, 0.32),
        (5.775, 14.875, 1.0),
        (8.375, 14.875, 0.32),
        (21.375, 14.875, 0.32),
        (23.975, 14.875, 1.0),
        (26.575, 14.875, 0.32),
        (5.775, 17.475, 0.62),
        (8.375, 17.475, 1.0),
        (10.975, 17.475, 1.0),
        (13.575, 17.475, 1.0),
        (16.175, 17.475, 1.0),
        (18.775, 17.475, 1.0),
        (21.375, 17.475, 1.0),
        (23.975, 17.475, 0.62),
        (8.375, 20.075, 0.32),
        (10.975, 20.075, 0.62),
        (13.575, 20.075, 1.0),
        (16.175, 20.075, 1.0),
        (18.775, 20.075, 0.62),
        (21.375, 20.075, 0.32),
    ]
    static let dot = 2.25, corner = 0.45
    /// Bounds of the dots, so the mark can be centred.
    static let minX = 3.175, maxX = 28.825, minY = 9.675, maxY = 22.325

    static func draw(in ctx: GraphicsContext, rect: CGRect, color: Color, emphasis: Double = 1) {
        let w = maxX - minX, h = maxY - minY
        let k = min(rect.width / w, rect.height / h)
        let ox = rect.midX - w * k / 2, oy = rect.midY - h * k / 2
        for (x, y, o) in dots {
            let r = CGRect(x: ox + (x - minX) * k, y: oy + (y - minY) * k, width: dot * k, height: dot * k)
            let a = o + (1 - o) * (1 - emphasis)  // emphasis 0 -> every dot full
            ctx.fill(Path(roundedRect: r, cornerRadius: corner * k), with: .color(color.opacity(a)))
        }
    }

    /// Template image for the menu bar (all dots solid while dictating).
    static func menuBarImage(active: Bool) -> NSImage {
        let size = NSSize(width: 22, height: 16)
        let img = NSImage(size: size, flipped: true) { rect in
            guard let cg = NSGraphicsContext.current?.cgContext else { return false }
            let w = maxX - minX, h = maxY - minY
            let k = min(rect.width / w, rect.height / h)
            let ox = rect.midX - w * k / 2, oy = rect.midY - h * k / 2
            for (x, y, o) in dots {
                cg.setFillColor(NSColor.black.withAlphaComponent(active ? 1 : max(o, 0.45)).cgColor)
                let r = CGRect(x: ox + (x - minX) * k, y: oy + (y - minY) * k, width: dot * k * 0.92, height: dot * k * 0.92)
                cg.addPath(CGPath(roundedRect: r, cornerWidth: corner * k, cornerHeight: corner * k, transform: nil))
                cg.fillPath()
            }
            return true
        }
        img.isTemplate = true
        return img
    }
}

/// The app icon in miniature (dark rounded square + mark).
struct LogoIcon: View {
    var body: some View {
        Canvas { ctx, size in
            let r = CGRect(origin: .zero, size: size)
            ctx.fill(Path(roundedRect: r, cornerRadius: size.width * 0.22),
                     with: .linearGradient(Gradient(colors: [Color(red: 0.149, green: 0.153, blue: 0.173),
                                                             Color(red: 0.047, green: 0.047, blue: 0.055)]),
                                           startPoint: .zero, endPoint: CGPoint(x: 0, y: size.height)))
            LogoMark.draw(in: ctx, rect: r.insetBy(dx: size.width * 0.2, dy: size.height * 0.2), color: .white)
        }
    }
}
