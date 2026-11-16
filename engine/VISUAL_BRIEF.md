# Optional local photograph analysis

This candidate describes visible colors, surface finish, focal decoration and recurring marks in one source photograph. It records each motif's location within the photographed face. The external-provider adapter keeps the palette, proposed finish, decoration family, placement and single/repeating distinction. Up to three source-facing physical structures are recorded separately from paint, together with the color and pattern of the visible body adjoining them. A handle, wire or recessed opening in one view does not establish a second copy on the reverse. Free-text motif labels stay on the computer. The unseen side and the physical material remain unknown.

Use Windows x64 and Python 3.11 with Pillow. Allow approximately 2.4 GB of project storage for the pinned runtime and at least 3.6 GiB available RAM per active analysis. A shared machine can need a larger reserve. Runtime data is installed in the directory you name. The default analysis uses three CPU threads, no GPU and no account. Select an idle slot when another generator is running.

From the repository root, install and recheck the files.

~~~powershell
python -m engine.visual_brief_setup .runtime/visual-brief
~~~

The installer downloads the pinned Windows llama.cpp distribution from GitHub and two public GGUF weight files from Hugging Face. For an offline installation, supply the directory containing these three original files with the --assets-dir option. All native DLLs, the executable and both model files are verified. Initial setup is the only operation that downloads files.

Run one photograph and use a different output directory for each source.

~~~powershell
python -m engine.visual_brief photo.jpg result/appearance-brief --runtime .runtime/visual-brief --threads 3
~~~

The fixed prompt and JSON grammar use Qwen3-VL-2B-Instruct Q8_0, seed 61732, temperature zero and at most 512 visual tokens. The command writes the validated model response to observation.json and records its source, recipe and run time in brief.json. Repeating the same request validates and returns the saved result. A different photo or recipe requires another directory. Rejected or truncated model output leaves this stage incomplete. No per-subject prompt or object-specific fallback is used.

Call analyze_photo(image, output_dir, runtime=..., threads=3) from Python. Pass its result and the source SHA-256 to provider_observations(report, source_sha256) to obtain closed-enumeration data for a guide request. A receiving adapter must call validate_provider_observations(payload, source_sha256) before using those fields, respect existing image-service consent, and bind the canonical payload digest into the request identity. Local analysis by itself sends no image or observations. A single view can support several plausible material or motif readings. The proposed finish and layout are interpretive. Preserving source pixels and evaluating other angles remain necessary.

The repository license covers this adapter. Qwen declares Apache-2.0 for the underlying model. GGUF files are distributed by ggml-org and llama.cpp is separately licensed under MIT. The installer pins upstream assets and users download them directly. [Original Qwen model](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) · [GGUF files](https://huggingface.co/ggml-org/Qwen3-VL-2B-Instruct-GGUF/tree/ea6a11058182570be6436b9a2e4ee7f7b49f908d) · [llama.cpp build](https://github.com/ggml-org/llama.cpp/releases/tag/b12420) · [llama.cpp license](https://github.com/ggml-org/llama.cpp/blob/master/LICENSE)
