import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @State private var models: [URL] = []
    @State private var selected: URL?
    @State private var prompt = "Hello! Tell me one fun fact about the moon."
    @State private var status = "Положи .gguf-файл в папку приложения (Файлы → На iPhone → PocketFit) или импортируй кнопкой."
    @State private var result: BenchResult?
    @State private var isRunning = false
    @State private var showImporter = false

    private var documentsURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    var body: some View {
        NavigationView {
            Form {
                Section("Модель") {
                    if models.isEmpty {
                        Text("Нет .gguf файлов").foregroundColor(.secondary)
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
                    Button("Импортировать .gguf") { showImporter = true }
                    Button("Обновить список") { refreshModels() }
                }

                Section("Промпт") {
                    TextField("Промпт", text: $prompt, axis: .vertical)
                }

                Section {
                    Button(isRunning ? "Идёт замер…" : "Запустить") { run() }
                        .disabled(isRunning || selected == nil)
                }

                Section("Результат") {
                    Text(status).font(.footnote).foregroundColor(.secondary)
                    if let r = result {
                        Text(String(format: "Скорость генерации: %.1f токенов/с", r.tokensPerSecond))
                        Text(String(format: "Время до первого токена: %.2f с", r.timeToFirstTokenSeconds))
                        Text(String(format: "Загрузка модели: %.2f с", r.loadSeconds))
                        Text("Токенов в промпте: \(r.promptTokens), сгенерировано: \(r.generatedTokens)")
                        Text(String(format: "Размер модели: %.0f МБ", r.modelSizeMB))
                        Text(r.modelDescription).font(.footnote)
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
            status = "Импорт не удался: \(error.localizedDescription)"
        case .success(let url):
            let access = url.startAccessingSecurityScopedResource()
            defer { if access { url.stopAccessingSecurityScopedResource() } }
            let dest = documentsURL.appendingPathComponent(url.lastPathComponent)
            do {
                if FileManager.default.fileExists(atPath: dest.path) {
                    try FileManager.default.removeItem(at: dest)
                }
                try FileManager.default.copyItem(at: url, to: dest)
                status = "Импортировано: \(dest.lastPathComponent)"
                refreshModels()
            } catch {
                status = "Копирование не удалось: \(error.localizedDescription)"
            }
        }
    }

    private func run() {
        guard let modelURL = selected else { return }
        isRunning = true
        result = nil
        status = "Загружаю модель и генерирую…"
        let path = modelURL.path
        let text = prompt
        Task.detached {
            do {
                let r = try LlamaRunner.run(modelPath: path, prompt: text, maxNewTokens: 64)
                await MainActor.run {
                    result = r
                    status = "Готово"
                    isRunning = false
                }
            } catch {
                await MainActor.run {
                    status = "Ошибка: \(error.localizedDescription)"
                    isRunning = false
                }
            }
        }
    }
}
