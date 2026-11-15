# Third-party components and model provenance

The diagnostic comparison of *Chasse with Christ in Majesty and the Lamb of God*, accession 17.190.523, includes two CC0 photographs from The Metropolitan Museum of Art and views rendered from the Studio trial. Only the three-quarter photograph was used for generation. [Museum collection record](https://www.metmuseum.org/art/collection/search/464494) · [Open Access policy](https://www.metmuseum.org/about-the-met/policies-and-documents/open-access)

The optional Pixal3D adapter sends a published CC0 photograph to a community-operated inference service. TencentARC publishes Pixal3D under MIT and documents separate third-party components. Studio distributes its client code. Operators of a self-hosted installation must also check the terms for its weights and components. [Upstream source and license](https://github.com/TencentARC/Pixal3D) · [Upstream notices](https://github.com/TencentARC/Pixal3D/blob/master/NOTICE) · [Space operator and transfer details](engine/PIXAL3D.md)

任意の CPU 形状エンジン TripoSG のコード、重み、同梱モジュールの個別条件、取得元・固定リビジョンは [TripoSG の第三者条件](engine/TRIPOSG.md#第三者の利用条件) に記載しています。これらの資材は各利用者の専用環境に取得します。

Inventory checked 2026-11-15 against the repository's setup and dependency declarations and the upstream sources linked below. This is an inventory, not a complete bill of materials or a replacement for the license texts required when distributing binaries. Dependencies and downloaded weights are runtime artifacts, not original project assets.

The optional local photograph reader downloads Qwen3-VL-2B-Instruct (Apache-2.0) and a pinned llama.cpp build (MIT). The ChatGPT observation and paint modes use the operator's Codex session and service terms. Model weights, account credentials, and service allowances are obtained separately by each operator. [Photo reader revisions and licenses](engine/VISUAL_BRIEF.md)

The optional Stable Fast 3D route sends the foreground to Stability AI's official Hugging Face Space. This repository distributes its own client. Stability AI provides the remote model under the Stability AI Community License, with conditions on commercial use. Each operator uses their own service allowance and checks the terms for their use of generated assets. [Model card](https://huggingface.co/stabilityai/stable-fast-3d) · [Upstream license](https://github.com/Stability-AI/stable-fast-3d/blob/main/LICENSE.md) · [Transfer and provider protocol](engine/SF3D.md)

## Engine and model assets

- [TripoSR code](https://github.com/VAST-AI-Research/TripoSR/tree/107cefdc244c39106fa830359024f6a2f1c78871): setup checks out commit `107cefdc244c39106fa830359024f6a2f1c78871`. Upstream [license](https://github.com/VAST-AI-Research/TripoSR/blob/107cefdc244c39106fa830359024f6a2f1c78871/LICENSE) is MIT. Preserve its copyright and license notice on redistribution.
- [TripoSR model](https://huggingface.co/stabilityai/TripoSR/tree/5b521936b01fbe1890f6f9baed0254ab6351c04a): setup downloads `config.yaml` and `model.ckpt` at revision `5b521936b01fbe1890f6f9baed0254ab6351c04a`; the model card declares MIT. Checkpoint SHA-256 verified by setup: `429e2c6b22a0923967459de24d67f05962b235f79cde6b032aa7ed2ffcd970ee`. Model license and code license are separate items.
- [DINO ViT-B/16 configuration](https://huggingface.co/facebook/dino-vitb16/blob/f205d5d8e640a89a2b8ef0369670dfc37cc07fc2/config.json): setup downloads only `config.json` at revision `f205d5d8e640a89a2b8ef0369670dfc37cc07fc2`; SHA-256 `b87c0270b97db085fd82cf114a761fd0f62ae7914fbd407c752a2260646b689c`. The [model card](https://huggingface.co/facebook/dino-vitb16) declares Apache-2.0. This inventory does not assert that separate DINO weights are downloaded.
- Optional background removal uses [rembg 2.0.57](https://github.com/danielgatis/rembg/tree/v2.0.57) (MIT) and its default U²-Net session. The [session source](https://github.com/danielgatis/rembg/blob/v2.0.57/rembg/sessions/u2net.py) retrieves `u2net.onnx` from a rembg release, checking MD5 `60024c5c889badc19c04ad937298a77b` unless disabled. Setup also verifies the official release asset against SHA-256 `8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491`. The downloaded weights' independent license and provenance are **unresolved**; do not assume the Python package's MIT license applies to the weights. Transparent inputs can bypass this optional path.
- UV unwrapping uses [xatlas Python bindings](https://github.com/mworchel/xatlas-python) 0.0.9, [MIT, copyright Markus Worchel](https://github.com/mworchel/xatlas-python/blob/main/LICENSE), and native [xatlas](https://github.com/jpcy/xatlas), [MIT, copyright Jonathan Young](https://github.com/jpcy/xatlas/blob/master/LICENSE). Preserve both applicable notices when redistributing a binary. The precise bundled native revision in the selected wheel remains unverified.

## Python packages

Direct pins from `engine/requirements.txt`, plus CPU PyTorch pins in `engine/setup.ps1`; links point to the exact upstream PyPI release metadata. License labels describe the projects' primary licenses; binary wheels can bundle additional components with separate notices.

| Package | Version | Declared primary license |
|---|---:|---|
| [numpy](https://pypi.org/project/numpy/1.26.4/) | 1.26.4 | BSD-3-Clause; bundled components have additional notices |
| [omegaconf](https://pypi.org/project/omegaconf/2.3.0/) | 2.3.0 | BSD |
| [pillow](https://pypi.org/project/Pillow/10.1.0/) | 10.1.0 | HPND |
| [einops](https://pypi.org/project/einops/0.7.0/) | 0.7.0 | MIT |
| [transformers](https://pypi.org/project/transformers/4.35.0/) | 4.35.0 | Apache-2.0 |
| [trimesh](https://pypi.org/project/trimesh/4.0.5/) | 4.0.5 | MIT |
| [huggingface-hub](https://pypi.org/project/huggingface-hub/0.17.3/) | 0.17.3 | Apache-2.0 |
| [scikit-image](https://pypi.org/project/scikit-image/0.22.0/) | 0.22.0 | BSD-3-Clause; see bundled notices |
| [xatlas](https://pypi.org/project/xatlas/0.0.9/) | 0.0.9 | MIT; see both notices above |
| [rembg](https://pypi.org/project/rembg/2.0.57/) | 2.0.57 | MIT; weights are separate |
| [onnxruntime](https://pypi.org/project/onnxruntime/1.18.1/) | 1.18.1 | MIT |
| [imageio](https://pypi.org/project/imageio/2.34.1/) | 2.34.1 | BSD-2-Clause |
| [torch](https://pypi.org/project/torch/2.2.2/) | 2.2.2 | BSD-3-Clause |
| [torchvision](https://pypi.org/project/torchvision/0.17.2/) | 0.17.2 | BSD |

Setup also uses [uv](https://github.com/astral-sh/uv) (MIT OR Apache-2.0) and downloads Python 3.11.15; Python is under the [PSF license](https://docs.python.org/3/license.html). Installers, Python distributions, wheels and transitive dependencies may carry their own notices.

## Viewer and validation tools

- [Three.js 0.180.0](https://github.com/mrdoob/three.js/tree/r180): viewer dependency; [MIT license](https://github.com/mrdoob/three.js/blob/r180/LICENSE).
- [Playwright 1.56.1](https://github.com/microsoft/playwright/tree/v1.56.1): development/browser test tooling (`@playwright/test`); [Apache-2.0](https://github.com/microsoft/playwright/blob/v1.56.1/LICENSE). Browser binaries are separate distributions with their own terms.
- [Khronos glTF Validator](https://github.com/KhronosGroup/glTF-Validator): optional externally supplied executable used by `quality/inspection.py`; [Apache-2.0](https://github.com/KhronosGroup/glTF-Validator/blob/main/LICENSE). No version is pinned in the repository; record the version and its bundled notices when used or redistributed.

## Outstanding distribution checks

1. Confirm independent rights and redistribution terms for the optional U²-Net weights before distributing or enabling that download in a public package.
2. Audit the actual resolved transitive packages, platform-specific wheels and bundled native libraries, browser binaries, and validator version. The direct-pin list above is not a complete dependency or notice inventory.
3. Collect and retain the applicable full license texts, copyright notices and any required NOTICE files with each distribution; this page alone does not fulfill every license's conditions. Recheck the model card, asset hashes, and licenses for the exact artifacts shipped.

## Photographs used for the general-purpose examples

The teapot photograph is an open-access image from the Cleveland Museum of Art, object 1940.10.a, by Davenport Pottery and Porcelain Factory. [Museum object and rights](https://www.clevelandart.org/art/1940.10.a) · [CC0](https://creativecommons.org/publicdomain/zero/1.0/).

The photograph of the Canon PowerShot SD1300 IS is by Infrogmation of New Orleans. The demos use its CC BY 3.0 license. Both examples derive the foreground, geometry, surface colors and comparison renders from a single photograph. Unseen surfaces are inferred. [Camera photograph](https://commons.wikimedia.org/wiki/File:CannonPowerShot1300Front.JPG) · [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) · [Model credits and comparisons](demo/general/).
