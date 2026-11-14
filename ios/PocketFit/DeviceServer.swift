import Foundation
import Network
import UIKit

// MARK: - Response models

struct DeviceInfo: Encodable {
    var identifier: String
    var iosVersion: String
    var physicalMemoryMb: Int
    var thermalState: String
    var lowPowerMode: Bool
    var charging: Bool
    var batteryLevel: Double
    var freeDiskMb: Int

    @MainActor static func current() -> DeviceInfo {
        UIDevice.current.isBatteryMonitoringEnabled = true
        var system = utsname()
        uname(&system)
        let machineSize = MemoryLayout.size(ofValue: system.machine)
        let identifier = withUnsafePointer(to: &system.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: machineSize) { String(cString: $0) }
        }
        return DeviceInfo(
            identifier: identifier,
            iosVersion: UIDevice.current.systemVersion,
            physicalMemoryMb: Int(ProcessInfo.processInfo.physicalMemory / 1_048_576),
            thermalState: thermalStateName(ProcessInfo.processInfo.thermalState),
            lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled,
            charging: DeviceServer.isCharging(),
            batteryLevel: Double(UIDevice.current.batteryLevel),
            freeDiskMb: Int(DeviceServer.freeDiskBytes() / 1_048_576)
        )
    }
}

private struct ErrorBody: Encodable {
    var error: String
}

private struct ModelEntry: Encodable {
    var name: String
    var sizeBytes: Int
}

private struct ModelsResponse: Encodable {
    var models: [ModelEntry]
}

private struct UploadResponse: Encodable {
    var name: String
    var sizeBytes: Int
}

private struct BenchSummary: Encodable {
    var minTokensPerSecond: Double
    var maxTokensPerSecond: Double
    var meanTokensPerSecond: Double
}

private struct BenchResponse: Encodable {
    var model: String
    var gpu: Bool
    var device: DeviceInfo
    var runs: [BenchResult]
    var summary: BenchSummary
}

struct BenchOptions {
    var useGPU = false
    var runs = 5
    var pauseSeconds: UInt64 = 10
    var tokens = 64
    var prompt = "Hello! Tell me one fun fact about the moon."

    /// Out-of-range values are rejected, not silently clamped, so the client always gets what it asked for.
    init(query: [String: String]) throws {
        if let value = query["gpu"] {
            guard ["0", "1", "true", "false"].contains(value) else {
                throw BenchOptionsError(message: "gpu must be 0 or 1")
            }
            useGPU = (value == "1" || value == "true")
        }
        runs = try Self.integer(query["runs"], name: "runs", default: runs, range: 1...20)
        pauseSeconds = UInt64(try Self.integer(query["pause"], name: "pause", default: Int(pauseSeconds), range: 0...120))
        tokens = try Self.integer(query["tokens"], name: "tokens", default: tokens, range: 1...512)
        if let value = query["prompt"], !value.isEmpty {
            guard value.count <= 2000 else {
                throw BenchOptionsError(message: "prompt must be at most 2000 characters")
            }
            prompt = value
        }
    }

    private static func integer(_ text: String?, name: String, default value: Int, range: ClosedRange<Int>) throws -> Int {
        guard let text = text else { return value }
        guard let number = Int(text), range.contains(number) else {
            throw BenchOptionsError(message: "\(name) must be between \(range.lowerBound) and \(range.upperBound)")
        }
        return number
    }
}

struct BenchOptionsError: Error {
    let message: String
}

// MARK: - Server

/// Small HTTP server that lets a computer on the same network send models to the app and run benchmarks.
/// Every request must carry the pairing token shown in the app.
final class DeviceServer {
    static let port: UInt16 = 8765
    static let serviceType = "_pocketfit._tcp"
    static let maxSessions = 4
    static let maxModelBytes = 8 * 1_073_741_824
    static let maxFailedAttempts = 5
    static let lockoutSeconds: TimeInterval = 60
    static let failureWindowSeconds: TimeInterval = 600

    let token: String
    let modelsDirectory: URL
    fileprivate let queue = DispatchQueue(label: "pocketfit.server")

    private let stateHandler: @Sendable (String) -> Void
    private var listener: NWListener?
    private var sessions: [ObjectIdentifier: HTTPSession] = [:]
    private var busy = false
    private var failures: [String: AuthFailures] = [:]

    private struct AuthFailures {
        var count = 0
        var lastFailure = Date.distantPast
        var blockedUntil: Date?
    }

    init(token: String, modelsDirectory: URL, stateHandler: @escaping @Sendable (String) -> Void) {
        self.token = token
        self.modelsDirectory = modelsDirectory
        self.stateHandler = stateHandler
    }

    func start() throws {
        removeStaleUploads()
        guard let port = NWEndpoint.Port(rawValue: Self.port) else { return }
        let parameters = NWParameters.tcp
        parameters.requiredInterfaceType = .wifi // never listen on cellular or other interfaces
        let listener = try NWListener(using: parameters, on: port)
        // Generic service name so the owner's device name is not broadcast on the network.
        listener.service = NWListener.Service(name: "PocketFit", type: Self.serviceType)
        listener.stateUpdateHandler = { [weak self] state in
            switch state {
            case .ready:
                self?.stateHandler("Listening on port \(Self.port)")
            case .failed(let error):
                self?.stateHandler("Server failed: \(error.localizedDescription)")
            default:
                break
            }
        }
        listener.newConnectionHandler = { [weak self] connection in
            self?.accept(connection)
        }
        self.listener = listener
        listener.start(queue: queue)
    }

    /// Deletes partial uploads left behind by a crash or a dropped connection.
    private func removeStaleUploads() {
        let directory = modelsDirectory
        queue.async {
            let keys: [URLResourceKey] = [.contentModificationDateKey]
            let urls = (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: keys)) ?? []
            for url in urls where url.lastPathComponent.hasSuffix(".gguf.part") {
                let modified = (try? url.resourceValues(forKeys: Set(keys)))?.contentModificationDate ?? Date.distantPast
                // An upload in progress keeps touching its file, so only old leftovers are removed.
                if Date().timeIntervalSince(modified) > 300 {
                    try? FileManager.default.removeItem(at: url)
                }
            }
        }
    }

    func stop() {
        listener?.cancel()
        listener = nil
        queue.async {
            for session in self.sessions.values { session.abort() }
            self.sessions.removeAll()
        }
    }

    private func accept(_ connection: NWConnection) {
        guard sessions.count < Self.maxSessions else {
            connection.cancel()
            return
        }
        let session = HTTPSession(connection: connection, server: self)
        let id = ObjectIdentifier(session)
        sessions[id] = session
        session.onClose = { [weak self] in
            self?.queue.async { self?.sessions[id] = nil }
        }
        session.start()
    }

    // MARK: Authentication. Call only on `queue`.

    /// Compares tokens without stopping at the first differing byte.
    static func tokensMatch(_ a: String, _ b: String) -> Bool {
        let x = Array(a.utf8)
        let y = Array(b.utf8)
        var difference = x.count ^ y.count
        for index in 0..<min(x.count, y.count) {
            difference |= Int(x[index] ^ y[index])
        }
        return difference == 0
    }

    fileprivate func isBlocked(_ host: String) -> Bool {
        if let until = failures[host]?.blockedUntil, until > Date() { return true }
        return false
    }

    fileprivate func recordFailure(_ host: String) {
        var entry = failures[host] ?? AuthFailures()
        if let until = entry.blockedUntil, until <= Date() { entry = AuthFailures() }
        // Old mistakes are forgotten, so a typo hours ago does not count towards a lockout.
        if Date().timeIntervalSince(entry.lastFailure) > Self.failureWindowSeconds { entry.count = 0 }
        entry.lastFailure = Date()
        entry.count += 1
        if entry.count >= Self.maxFailedAttempts {
            entry.blockedUntil = Date().addingTimeInterval(Self.lockoutSeconds)
            entry.count = 0
        }
        failures[host] = entry
    }

    fileprivate func recordSuccess(_ host: String) {
        failures[host] = nil
    }

    // MARK: Busy flag (one upload or benchmark at a time). Call only on `queue`.

    fileprivate func beginBusy() -> Bool {
        if busy { return false }
        busy = true
        return true
    }

    /// The screen stays awake for as long as "Accept connections" is on; the app owns that setting.
    fileprivate func releaseBusy() {
        busy = false
    }

    // MARK: Helpers

    static func safeModelName(_ raw: String) -> String? {
        let name = (raw as NSString).lastPathComponent
        guard !name.isEmpty, name == raw, !name.hasPrefix("."), name.lowercased().hasSuffix(".gguf") else { return nil }
        return name
    }

    fileprivate func modelURL(named raw: String) -> URL? {
        guard let name = Self.safeModelName(raw) else { return nil }
        return modelsDirectory.appendingPathComponent(name)
    }

    @MainActor static func isCharging() -> Bool {
        UIDevice.current.isBatteryMonitoringEnabled = true
        let state = UIDevice.current.batteryState
        return state == .charging || state == .full
    }

    static func freeDiskBytes() -> Int64 {
        let values = try? URL(fileURLWithPath: NSHomeDirectory())
            .resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return values?.volumeAvailableCapacityForImportantUsage ?? 0
    }

    fileprivate func listModels() -> [ModelEntry] {
        let urls = (try? FileManager.default.contentsOfDirectory(at: modelsDirectory, includingPropertiesForKeys: nil)) ?? []
        return urls
            .filter { $0.pathExtension.lowercased() == "gguf" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
            .map { url in
                let attributes = try? FileManager.default.attributesOfItem(atPath: url.path)
                let size = (attributes?[.size] as? NSNumber)?.intValue ?? 0
                return ModelEntry(name: url.lastPathComponent, sizeBytes: size)
            }
    }

    fileprivate func runBenchmark(path: URL, name: String, options: BenchOptions) async -> (Int, Data) {
        var results: [BenchResult] = []
        do {
            for index in 1...options.runs {
                if Task.isCancelled { return (499, Self.encode(ErrorBody(error: "Cancelled"))) }
                let charging = await MainActor.run { DeviceServer.isCharging() }
                let result = try await Task.detached(priority: .userInitiated) {
                    try LlamaRunner.run(
                        modelPath: path.path,
                        prompt: options.prompt,
                        maxNewTokens: options.tokens,
                        useGPU: options.useGPU,
                        charging: charging
                    )
                }.value
                results.append(result)
                if index < options.runs {
                    try? await Task.sleep(nanoseconds: options.pauseSeconds * 1_000_000_000)
                }
            }
        } catch {
            return (500, Self.encode(ErrorBody(error: error.localizedDescription)))
        }

        let speeds = results.map { $0.tokensPerSecond }
        let summary = BenchSummary(
            minTokensPerSecond: speeds.min() ?? 0,
            maxTokensPerSecond: speeds.max() ?? 0,
            meanTokensPerSecond: speeds.isEmpty ? 0 : speeds.reduce(0, +) / Double(speeds.count)
        )
        let device = await MainActor.run { DeviceInfo.current() }
        let response = BenchResponse(model: name, gpu: options.useGPU, device: device, runs: results, summary: summary)
        return (200, Self.encode(response))
    }

    fileprivate static func encode<T: Encodable>(_ value: T) -> Data {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return (try? encoder.encode(value)) ?? Data("{}".utf8)
    }
}

// MARK: - HTTP session

private struct HTTPRequest {
    var method: String
    var path: String
    var query: [String: String]
    var headers: [String: String]
}

private final class Upload {
    let name: String
    let expected: Int
    var received = 0
    var head = Data() // first bytes of the body, checked for the GGUF magic
    let handle: FileHandle
    let temp: URL
    let final: URL

    init(name: String, expected: Int, handle: FileHandle, temp: URL, final: URL) {
        self.name = name
        self.expected = expected
        self.handle = handle
        self.temp = temp
        self.final = final
    }
}

/// Handles one HTTP/1.1 connection: parse the request, stream upload bodies to disk, send one JSON response.
private final class HTTPSession {
    var onClose: (() -> Void)?

    private let connection: NWConnection
    private let server: DeviceServer
    private var buffer = Data()
    private var request: HTTPRequest?
    private var upload: Upload?
    private var holdsBusy = false
    private var benchTask: Task<Void, Never>?
    private var finished = false
    private var closed = false
    private var lastActivity = Date()
    private var idleLimit: TimeInterval? = 30

    private lazy var host: String = {
        if case let .hostPort(host, _) = connection.endpoint { return "\(host)" }
        return "unknown"
    }()

    init(connection: NWConnection, server: DeviceServer) {
        self.connection = connection
        self.server = server
    }

    /// Closes connections that stall: slow header senders, stuck uploads. Disabled while a benchmark runs.
    private func scheduleIdleCheck() {
        server.queue.asyncAfter(deadline: .now() + 5) { [weak self] in
            guard let self = self, !self.closed else { return }
            if let limit = self.idleLimit, Date().timeIntervalSince(self.lastActivity) > limit {
                self.close()
                return
            }
            self.scheduleIdleCheck()
        }
    }

    func start() {
        scheduleIdleCheck()
        connection.stateUpdateHandler = { [weak self] state in
            switch state {
            case .failed, .cancelled:
                self?.close()
            default:
                break
            }
        }
        connection.start(queue: server.queue)
        receive()
    }

    func abort() {
        close()
    }

    // MARK: Receiving

    private func receive() {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 1 << 20) { [weak self] data, _, isComplete, error in
            guard let self = self, !self.closed else { return }
            if let data = data, !data.isEmpty {
                self.consume(data)
            }
            if error != nil {
                self.close()
                return
            }
            if isComplete && !self.finished {
                // The client closed its sending side. Keep the connection only while a response is still pending.
                if self.request == nil || self.upload != nil { self.close() }
                return
            }
            if !self.finished && !self.closed {
                self.receive()
            }
        }
    }

    private func consume(_ data: Data) {
        lastActivity = Date()
        if request != nil {
            handleBody(data)
            return
        }
        buffer.append(data)
        guard let range = buffer.range(of: Data("\r\n\r\n".utf8)) else {
            if buffer.count > 16_384 { respondError(431, "Request headers too large") }
            return
        }
        let head = buffer.subdata(in: buffer.startIndex..<range.lowerBound)
        let rest = buffer.subdata(in: range.upperBound..<buffer.endIndex)
        buffer.removeAll()
        guard head.count <= 16_384 else {
            respondError(431, "Request headers too large")
            return
        }
        guard let parsed = parse(head) else {
            respondError(400, "Malformed request")
            return
        }
        guard parsed.headers["transfer-encoding"] == nil else {
            respondError(400, "Chunked transfer encoding is not supported")
            return
        }
        request = parsed
        route(parsed, firstChunk: rest)
    }

    private func parse(_ head: Data) -> HTTPRequest? {
        let text = String(decoding: head, as: UTF8.self)
        let lines = text.components(separatedBy: "\r\n")
        guard let first = lines.first else { return nil }
        let parts = first.split(separator: " ")
        guard parts.count >= 2, let components = URLComponents(string: String(parts[1])) else { return nil }
        var query: [String: String] = [:]
        for item in components.queryItems ?? [] {
            query[item.name] = item.value ?? ""
        }
        var headers: [String: String] = [:]
        for line in lines.dropFirst() {
            guard let colon = line.firstIndex(of: ":") else { continue }
            let name = line[..<colon].trimmingCharacters(in: .whitespaces).lowercased()
            let value = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
            headers[name] = value
        }
        return HTTPRequest(method: String(parts[0]), path: components.path, query: query, headers: headers)
    }

    // MARK: Routing

    private func route(_ req: HTTPRequest, firstChunk: Data) {
        if server.isBlocked(host) {
            respondError(429, "Too many failed attempts, try again later")
            return
        }
        guard DeviceServer.tokensMatch(req.headers["x-pocketfit-token"] ?? "", server.token) else {
            server.recordFailure(host)
            respondError(401, "Invalid or missing token")
            return
        }
        server.recordSuccess(host)
        switch (req.method, req.path) {
        case ("GET", "/info"):
            Task { [weak self] in
                let info = await MainActor.run { DeviceInfo.current() }
                self?.server.queue.async { self?.respondJSON(200, info) }
            }
        case ("GET", "/models"):
            respondJSON(200, ModelsResponse(models: server.listModels()))
        case ("POST", "/bench"):
            startBenchmark(req)
        case ("PUT", _) where req.path.hasPrefix("/models/"):
            startUpload(req, firstChunk: firstChunk)
        case ("DELETE", _) where req.path.hasPrefix("/models/"):
            deleteModel(req)
        default:
            respondError(404, "Not found")
        }
    }

    private func startBenchmark(_ req: HTTPRequest) {
        guard let name = req.query["model"], let url = server.modelURL(named: name) else {
            respondError(400, "Missing or invalid model name")
            return
        }
        guard FileManager.default.fileExists(atPath: url.path) else {
            respondError(404, "Model not found on device")
            return
        }
        let options: BenchOptions
        do {
            options = try BenchOptions(query: req.query)
        } catch let error as BenchOptionsError {
            respondError(400, error.message)
            return
        } catch {
            respondError(400, "Invalid options")
            return
        }
        guard server.beginBusy() else {
            respondError(409, "Device is busy")
            return
        }
        idleLimit = nil // the client legitimately waits for the whole benchmark
        let server = self.server
        // Cancelled in close(), so a disconnect or "Accept connections" off stops the benchmark between runs.
        benchTask = Task { [weak self] in
            let (status, body) = await server.runBenchmark(path: url, name: name, options: options)
            server.queue.async {
                server.releaseBusy()
                self?.respond(status: status, body: body)
            }
        }
    }

    private func deleteModel(_ req: HTTPRequest) {
        let raw = String(req.path.dropFirst("/models/".count))
        guard let url = server.modelURL(named: raw) else {
            respondError(400, "Invalid model name")
            return
        }
        guard server.beginBusy() else {
            respondError(409, "Device is busy")
            return
        }
        defer { server.releaseBusy() }
        do {
            try FileManager.default.removeItem(at: url)
            respondJSON(200, ErrorBody(error: ""))
        } catch {
            respondError(404, "Model not found on device")
        }
    }

    // MARK: Upload

    private func startUpload(_ req: HTTPRequest, firstChunk: Data) {
        let raw = String(req.path.dropFirst("/models/".count))
        guard let final = server.modelURL(named: raw) else {
            respondError(400, "Model name must be a plain .gguf file name")
            return
        }
        guard let length = Int(req.headers["content-length"] ?? ""), length >= 0 else {
            respondError(411, "Content-Length required")
            return
        }
        guard length >= 4 else {
            respondError(400, "File is too small to be a GGUF model")
            return
        }
        guard length <= DeviceServer.maxModelBytes else {
            respondError(413, "Model is larger than the 8 GB limit")
            return
        }
        idleLimit = 60
        if Int64(length) > DeviceServer.freeDiskBytes() - 200 * 1_048_576 {
            respondError(507, "Not enough free storage on the device")
            return
        }
        guard server.beginBusy() else {
            respondError(409, "Device is busy")
            return
        }
        holdsBusy = true

        let temp = server.modelsDirectory.appendingPathComponent(final.lastPathComponent + ".part")
        FileManager.default.createFile(atPath: temp.path, contents: nil)
        guard let handle = try? FileHandle(forWritingTo: temp) else {
            respondError(500, "Cannot create the model file")
            return
        }
        upload = Upload(name: final.lastPathComponent, expected: length, handle: handle, temp: temp, final: final)
        if !firstChunk.isEmpty {
            handleBody(firstChunk)
        }
    }

    private func handleBody(_ data: Data) {
        guard let up = upload, !finished else { return }
        let remaining = up.expected - up.received
        let chunk = data.count > remaining ? data.prefix(remaining) : data
        if up.head.count < 4 {
            up.head.append(chunk.prefix(4 - up.head.count))
            if up.head.count == 4 && up.head != Data("GGUF".utf8) {
                respondError(400, "File is not a GGUF model")
                return
            }
        }
        do {
            try up.handle.write(contentsOf: chunk)
        } catch {
            respondError(500, "Write failed: \(error.localizedDescription)")
            return
        }
        up.received += chunk.count
        if up.received >= up.expected {
            finishUpload()
        }
    }

    private func finishUpload() {
        guard let up = upload else { return }
        try? up.handle.close()
        do {
            if FileManager.default.fileExists(atPath: up.final.path) {
                // Atomic replace: the old model is kept if this fails.
                _ = try FileManager.default.replaceItemAt(up.final, withItemAt: up.temp)
            } else {
                try FileManager.default.moveItem(at: up.temp, to: up.final)
            }
        } catch {
            respondError(500, "Could not store the model: \(error.localizedDescription)")
            return
        }
        upload = nil
        respondJSON(201, UploadResponse(name: up.name, sizeBytes: up.received))
    }

    // MARK: Responding

    private func respondError(_ status: Int, _ message: String) {
        respondJSON(status, ErrorBody(error: message))
    }

    private func respondJSON<T: Encodable>(_ status: Int, _ value: T) {
        respond(status: status, body: DeviceServer.encode(value))
    }

    fileprivate func respond(status: Int, body: Data) {
        guard !finished, !closed else { return }
        finished = true
        // Free the device before the client can see the response, so its next request is not refused as busy.
        if holdsBusy {
            holdsBusy = false
            server.releaseBusy()
        }
        let head = "HTTP/1.1 \(status) \(reason(status))\r\nContent-Type: application/json\r\nContent-Length: \(body.count)\r\nConnection: close\r\n\r\n"
        var out = Data(head.utf8)
        out.append(body)
        connection.send(content: out, completion: .contentProcessed { [weak self] _ in
            self?.close()
        })
    }

    private func reason(_ status: Int) -> String {
        switch status {
        case 200: return "OK"
        case 201: return "Created"
        case 400: return "Bad Request"
        case 401: return "Unauthorized"
        case 404: return "Not Found"
        case 409: return "Conflict"
        case 411: return "Length Required"
        case 413: return "Payload Too Large"
        case 429: return "Too Many Requests"
        case 431: return "Request Header Fields Too Large"
        case 507: return "Insufficient Storage"
        default: return "Internal Server Error"
        }
    }

    private func close() {
        guard !closed else { return }
        closed = true
        benchTask?.cancel()
        connection.cancel()
        if let up = upload {
            try? up.handle.close()
            try? FileManager.default.removeItem(at: up.temp)
            upload = nil
        }
        if holdsBusy {
            holdsBusy = false
            server.releaseBusy()
        }
        onClose?()
    }
}
