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
                        Text("Thermal state at start: \(r.thermalState)")
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
        Task.detached {
            do {
                let r = try LlamaRunner.run(modelPath: path, prompt: text, maxNewTokens: 64, useGPU: gpu)
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
}
