import Foundation
import PDFKit
import Vision
import AppKit
let url = URL(fileURLWithPath: CommandLine.arguments[1])
guard let doc = PDFDocument(url: url) else { fatalError("Cannot open PDF") }
var pages: [String] = []
for index in 0..<doc.pageCount {
    guard let page = doc.page(at:index) else { fatalError("Missing page") }
    let bounds = page.bounds(for:.mediaBox)
    let image = page.thumbnail(of:NSSize(width:bounds.width*3, height:bounds.height*3), for:.mediaBox)
    guard let cgImage = image.cgImage(forProposedRect:nil, context:nil, hints:nil) else { fatalError("Cannot rasterize") }
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.recognitionLanguages = ["en-US"]
    request.usesLanguageCorrection = true
    try VNImageRequestHandler(cgImage:cgImage).perform([request])
    pages.append((request.results ?? []).compactMap { $0.topCandidates(1).first?.string }.joined(separator:"\n"))
}
let data = try JSONSerialization.data(withJSONObject:pages)
FileHandle.standardOutput.write(data)
