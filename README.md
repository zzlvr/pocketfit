# PocketFit

One command from a fine-tuned model to a model that actually runs well on your phone.

> Status: week 1 of 4. Right now this repo only builds a minimal iPhone app that runs a GGUF model with llama.cpp on Metal and prints tokens/s. The auto-tuner is not written yet.

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

- [ ] Week 1 (in progress): build pipeline (GitHub Actions on macOS, no Mac needed), minimal on-device runner
- [ ] Week 2: send a model over Wi-Fi from a CLI, get metrics back from the phone
- [ ] Week 3: auto-selection of quantization, quality measurement, report
- [ ] Week 4: public benchmark table, write-up, release

## Build the iPhone app without a Mac

The workflow `.github/workflows/ios-build.yml` builds an **unsigned** IPA on a GitHub macOS runner. Download the `PocketFit-ipa` artifact from the Actions tab, then install it with [Sideloadly](https://sideloadly.io/) using a free Apple ID. Free-account apps expire after 7 days and must be reinstalled.

Put a `.gguf` model into the app folder (Files app, On My iPhone, PocketFit) or use the Import button, then tap Run.

## License

MIT
