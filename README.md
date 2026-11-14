# PocketFit

One command from a fine-tuned model to a model that actually runs well on your phone.

## The problem

Getting your own model onto a phone is a pile of manual steps: pick a format, guess a quantization, build native libraries, test on one phone, then find out it crashes or crawls on another. Existing tools (llama.cpp, MLC-LLM, ExecuTorch, Cactus) each cover part of it. Nobody picks the quantization and runtime for you by measuring on your real device.

## The idea

```
pocketfit ship ./my-model --devices iphone
```

1. Candidate quantizations are filtered by the device's memory.
2. Each candidate is run on the real device; speed, memory and thermal state are recorded.
3. Quality loss is measured against the full-precision model.
4. The best option under your constraints (for example `>= 10 tok/s` and `< 2 %` quality loss) is picked, and you get a package plus a report.

## Roadmap

- [x] iPhone runner: load a GGUF model with llama.cpp and report tokens/s
- [x] Send a model over Wi-Fi from a CLI and get metrics back from the phone
- [ ] Automatic quantization selection with quality measurement and a report
- [ ] Public benchmark table

## Build the iPhone app without a Mac

The workflow `.github/workflows/ios-build.yml` builds an **unsigned** IPA on a GitHub macOS runner. Download the `PocketFit-ipa` artifact from the Actions tab, then install it with [Sideloadly](https://sideloadly.io/) using a free Apple ID. Free-account apps expire after 7 days and must be reinstalled.

In the app, tap Import to pick a `.gguf` model, then tap Run.

## License

MIT
