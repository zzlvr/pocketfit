# AGENTS.md

Guidance for coding agents working in this repository.

## What this project is

PocketFit aims to pick the best quantization and runtime for a model by measuring it on the real device. See `README.md` for the idea and the roadmap.

The repository currently contains an iPhone app (SwiftUI + llama.cpp) that loads a GGUF model, generates 64 tokens greedily and reports tokens/s, time to first token and model load time, plus a GitHub Actions workflow that builds the app.

## Layout

```
.github/workflows/ios-build.yml   CI: XcodeGen, llama.cpp xcframework download, unsigned xcodebuild, IPA packaging
ios/project.yml                   XcodeGen project description (iOS 16+, embeds llama.xcframework)
ios/PocketFit/PocketFitApp.swift  app entry point
ios/PocketFit/ContentView.swift   UI: model list, import, prompt, run, results
ios/PocketFit/LlamaRunner.swift   model loading, tokenization, generation, measurements
```

Not committed, produced by the workflow: `ios/Frameworks/`, `ios/PocketFit.xcodeproj/`, `ios/PocketFit/Info.plist`, `out/`, `*.gguf`.

## Building

Xcode is only available on macOS, so the app is built by the `ios-build` workflow on a GitHub macOS runner. It produces an unsigned IPA as the `PocketFit-ipa` artifact. On a Mac, follow the same steps as the workflow: fetch the llama.cpp xcframework for the pinned tag, run `xcodegen generate` in `ios/`, then `xcodebuild` for `iphoneos`.

## Installing on a device

Install the IPA with [Sideloadly](https://sideloadly.io/) and a free Apple ID. On Windows, Sideloadly needs the web versions of iTunes and iCloud, not the Microsoft Store ones. Free-account apps expire after 7 days. Enable Developer Mode on the iPhone, trust the developer profile, then use the Import button in the app to select a `.gguf` file.

## Conventions

- llama.cpp is pinned with `LLAMA_TAG` in the workflow. Before changing the tag, compare `llama.h` from that release with the Swift code, since the API changes often. Use the current API (`llama_model_load_from_file`, `llama_init_from_model`, ...), not the deprecated one.
- Everything in the repository is in English: code, comments, UI strings, docs.
- Commit messages are one short line describing what changed.
- Keep the app in the foreground while measuring, and record the thermal state with each result, since throttling distorts timings.
- Do not commit secrets, Apple IDs, passwords or personal data.
