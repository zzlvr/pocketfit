import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @State private var models: [URL] = []
    @State private var selected: URL?
    @State private var prompt = "Hello! Tell me one fun fact about the moon."
    @State private var status = "Import a .gguf model with the button below."
    @State private var result: BenchResult?
    @State private var isRunning = false
    @State private var showImporter = false
    @State private var useGPU = true
    @State private var benchRuns: [BenchResult] = []
    private let benchCount = 5
    private let benchPauseSeconds: UInt64 = 10

    private var documentsURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    var body: some View {
        NavigationView {
            Form {
                Section("Model") {
                    if models.isEmpty {
                        Text("No .gguf files").foregroundColor(.secondary)
                    }
                    ForEach(models, id: \.self) { url in
                        Button {
                            selected = url
                        } label: {
                            HStack {
                                Text(url.lastPathComponent).lineLimit(1)
                                Spacer()
                                if selected == url { Image(systemName: "checkmark") }
                            }
                        }
                    }
                    Button("Import .gguf") { showImporter = true }
                    Button("Refresh list") { refreshModels() }
                }

                Section("Prompt") {
                    TextField("Prompt", text: $prompt, axis: .vertical)
                }

                Section {
                    Toggle("GPU (Metal)", isOn: $useGPU)
                    Button(isRunning ? "Running..." : "Run") { run() }
                        .disabled(isRunning || selected == nil)
                    Button("Benchmark (\(benchCount) runs)") { runBenchmark() }
                        .disabled(isRunning || selected == nil)
                }

                if !benchRuns.isEmpty {
                    Section("Benchmark") {
                        ForEach(Array(benchRuns.enumerated()), id: \.offset) { index, r in
                            Text("#\(index + 1): \(String(format: "%.1f", r.tokensPerSecond)) tokens/s, thermal \(r.thermalState) -> \(r.thermalStateEnd), charging: \(r.charging ? "yes" : "no")")
                                .font(.footnote)
                        }
                        let speeds = benchRuns.map { $0.tokensPerSecond }
                        if let lo = speeds.min(), let hi = speeds.max() {
                            Text("Min \(String(format: "%.1f", lo)), max \(String(format: "%.1f", hi)), mean \(String(format: "%.1f", speeds.reduce(0, +) / Double(speeds.count))) tokens/s")
                        }
                    }
                }

                Section("Result") {
                    Text(status).font(.footnote).foregroundColor(.secondary)
                    if let r = result {
                        Text(String(format: "Generation speed: %.1f tokens/s", r.tokensPerSecond))
                        Text(String(format: "Time to first token: %.2f s", r.timeToFirstTokenSeconds))
                        Text(String(format: "Model load time: %.2f s", r.loadSeconds))
                        Text("Prompt tokens: \(r.promptTokens), generated: \(r.generatedTokens)")
                        Text(String(format: "Model size: %.0f MB", r.modelSizeMB))
                        Text(r.modelDescription).font(.footnote)
                        Text("GPU requested: \(r.gpuRequested ? "yes" : "no"), GPU offload supported: \(r.gpuOffloadSupported ? "yes" : "no")")
                        Text("Thermal state: \(r.thermalState) at start, \(r.thermalStateEnd) at end")
                        Text("Charging: \(r.charging ? "yes" : "no")")
                        Text(r.systemInfo).font(.caption2).foregroundColor(.secondary)
                        Text(r.text).font(.body)
                    }
                }
            }
            .navigationTitle("PocketFit")
        }
        .onAppear(perform: refreshModels)
        .fileImporter(isPresented: $showImporter, allowedContentTypes: [.data]) { importResult in
            importModel(importResult)
        }
    }

    private func refreshModels() {
        let items = (try? FileManager.default.contentsOfDirectory(at: documentsURL, includingPropertiesForKeys: nil)) ?? []
        models = items.filter { $0.pathExtension.lowercased() == "gguf" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        if selected == nil { selected = models.first }
    }

    private func importModel(_ importResult: Result<URL, Error>) {
        switch importResult {
        case .failure(let error):
            status = "Import failed: \(error.localizedDescription)"
        case .success(let url):
            let access = url.startAccessingSecurityScopedResource()
            defer { if access { url.stopAccessingSecurityScopedResource() } }
            let dest = documentsURL.appendingPathComponent(url.lastPathComponent)
            do {
                if FileManager.default.fileExists(atPath: dest.path) {
                    try FileManager.default.removeItem(at: dest)
                }
                try FileManager.default.copyItem(at: url, to: dest)
                status = "Imported: \(dest.lastPathComponent)"
                refreshModels()
            } catch {
                status = "Copy failed: \(error.localizedDescription)"
            }
        }
    }

    private func run() {
        guard let modelURL = selected else { return }
        isRunning = true
        result = nil
        status = "Loading model and generating..."
        let path = modelURL.path
        let text = prompt
        let gpu = useGPU
        let charging = isCharging()
        Task.detached {
            do {
                let r = try LlamaRunner.run(modelPath: path, prompt: text, maxNewTokens: 64, useGPU: gpu, charging: charging)
                await MainActor.run {
                    result = r
                    status = "Done"
                    isRunning = false
                }
            } catch {
                await MainActor.run {
                    status = "Error: \(error.localizedDescription)"
                    isRunning = false
                }
            }
        }
    }

    private func runBenchmark() {
        guard let modelURL = selected else { return }
        isRunning = true
        result = nil
        benchRuns = []
        let path = modelURL.path
        let text = prompt
        let gpu = useGPU
        let count = benchCount
        let pause = benchPauseSeconds
        Task.detached {
            do {
                for i in 1...count {
                    await MainActor.run { status = "Benchmark run \(i) of \(count)..." }
                    let charging = await MainActor.run { isCharging() }
                    let r = try LlamaRunner.run(modelPath: path, prompt: text, maxNewTokens: 64, useGPU: gpu, charging: charging)
                    await MainActor.run { benchRuns.append(r) }
                    if i < count {
                        await MainActor.run { status = "Pausing \(pause) s before run \(i + 1)..." }
                        try? await Task.sleep(nanoseconds: pause * 1_000_000_000)
                    }
                }
                await MainActor.run {
                    status = "Benchmark done"
                    isRunning = false
                }
            } catch {
                await MainActor.run {
                    status = "Error: \(error.localizedDescription)"
                    isRunning = false
                }
            }
        }
    }

    private func isCharging() -> Bool {
        UIDevice.current.isBatteryMonitoringEnabled = true
        let state = UIDevice.current.batteryState
        return state == .charging || state == .full
    }
}
