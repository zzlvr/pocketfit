import Foundation
import llama

struct BenchResult {
    var text: String
    var promptTokens: Int
    var generatedTokens: Int
    var loadSeconds: Double
    var timeToFirstTokenSeconds: Double
    var tokensPerSecond: Double
    var modelDescription: String
    var modelSizeMB: Double
}

enum LlamaError: Error, LocalizedError {
    case modelLoadFailed
    case contextFailed
    case tokenizeFailed
    case decodeFailed(Int32)

    var errorDescription: String? {
        switch self {
        case .modelLoadFailed: return "Не удалось загрузить модель (файл повреждён или не хватает памяти)"
        case .contextFailed: return "Не удалось создать контекст (вероятно, не хватает памяти)"
        case .tokenizeFailed: return "Не удалось разбить промпт на токены"
        case .decodeFailed(let code): return "Ошибка декодирования, код \(code)"
        }
    }
}

/// Минимальный запуск llama.cpp на устройстве: загрузка модели, жадная генерация, замер скорости.
enum LlamaRunner {
    static func run(modelPath: String, prompt: String, maxNewTokens: Int, contextSize: Int = 2048) throws -> BenchResult {
        llama_backend_init()
        defer { llama_backend_free() }

        let loadStart = CFAbsoluteTimeGetCurrent()

        var modelParams = llama_model_default_params()
        modelParams.n_gpu_layers = 999 // все слои на Metal GPU

        guard let model = llama_model_load_from_file(modelPath, modelParams) else {
            throw LlamaError.modelLoadFailed
        }
        defer { llama_model_free(model) }

        var ctxParams = llama_context_default_params()
        ctxParams.n_ctx = UInt32(contextSize)
        ctxParams.n_batch = UInt32(contextSize)

        guard let ctx = llama_init_from_model(model, ctxParams) else {
            throw LlamaError.contextFailed
        }
        defer { llama_free(ctx) }

        let loadSeconds = CFAbsoluteTimeGetCurrent() - loadStart

        var descBuf = [CChar](repeating: 0, count: 256)
        _ = llama_model_desc(model, &descBuf, descBuf.count)
        let modelDescription = String(cString: descBuf)
        let modelSizeMB = Double(llama_model_size(model)) / 1_048_576.0

        guard let vocab = llama_model_get_vocab(model) else { throw LlamaError.modelLoadFailed }

        // Токенизация промпта
        var tokens = [llama_token](repeating: 0, count: contextSize)
        let promptByteCount = Int32(prompt.utf8.count)
        let tokenCount = llama_tokenize(vocab, prompt, promptByteCount, &tokens, Int32(tokens.count), true, true)
        guard tokenCount > 0 else { throw LlamaError.tokenizeFailed }
        tokens = Array(tokens.prefix(Int(tokenCount)))

        let sampler = llama_sampler_chain_init(llama_sampler_chain_default_params())
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy())
        defer { llama_sampler_free(sampler) }

        // Обработка промпта
        let promptStart = CFAbsoluteTimeGetCurrent()
        let promptRC = tokens.withUnsafeMutableBufferPointer { buf -> Int32 in
            let batch = llama_batch_get_one(buf.baseAddress, Int32(buf.count))
            return llama_decode(ctx, batch)
        }
        guard promptRC == 0 else { throw LlamaError.decodeFailed(promptRC) }

        var next = llama_sampler_sample(sampler, ctx, -1)
        let firstTokenTime = CFAbsoluteTimeGetCurrent()
        let ttft = firstTokenTime - promptStart

        // Генерация
        var outBytes = [UInt8]()
        var generated = 0
        var pieceBuf = [CChar](repeating: 0, count: 256)

        while generated < maxNewTokens && !llama_vocab_is_eog(vocab, next) {
            let n = llama_token_to_piece(vocab, next, &pieceBuf, Int32(pieceBuf.count), 0, true)
            if n > 0 {
                outBytes.append(contentsOf: pieceBuf.prefix(Int(n)).map { UInt8(bitPattern: $0) })
            }
            generated += 1

            var token = next
            let rc = withUnsafeMutablePointer(to: &token) { ptr -> Int32 in
                llama_decode(ctx, llama_batch_get_one(ptr, 1))
            }
            guard rc == 0 else { throw LlamaError.decodeFailed(rc) }
            next = llama_sampler_sample(sampler, ctx, -1)
        }

        let genSeconds = CFAbsoluteTimeGetCurrent() - firstTokenTime
        let tps = genSeconds > 0 ? Double(generated) / genSeconds : 0

        return BenchResult(
            text: String(decoding: outBytes, as: UTF8.self),
            promptTokens: tokens.count,
            generatedTokens: generated,
            loadSeconds: loadSeconds,
            timeToFirstTokenSeconds: ttft,
            tokensPerSecond: tps,
            modelDescription: modelDescription,
            modelSizeMB: modelSizeMB
        )
    }
}
