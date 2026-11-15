# Paint atlas review

`paint_audit.py` is a conservative aid to human inspection, never an appearance pass. It reads an embedded GLB base-color image and reports connected off-palette components inside manually selected pixel rectangles. For glTF baseColorFactor it multiplies texture sRGB in linear space and converts back to sRGB before comparison. The report records the GLB SHA-256. A result is always `review` or `unmeasured`; `appearance` remains `pending_human_review`.

Create a JSON profile outside the repository, selecting a unique material name or an explicit material index for unnamed materials, and an atlas rectangle known by inspection to represent the desired part. Palette entries must include every intended color, including stripes. Example:

```json
{"regions":[{"name":"body-paint","material_index":0,"rect":[0,0,128,128],"palette":[[220,180,100],[40,40,40]],"tolerance":12,"max_island_pixels":50}]}
```

Coordinates are pixels, top-left origin, `[left, top, right, bottom)`; use the actual texture dimensions. `tolerance` is Euclidean RGB distance in 0–255 sRGB units. Four-connected components no larger than `max_island_pixels` are flagged as small; larger off-palette components are also reported, never silently accepted. Edge-touching components may be clipped by the chosen rectangle. Inspect both the atlas and hash-matched material and clay renders before judging any finding.

Run with the setup environment's Pillow:

```powershell
.runtime\venv\Scripts\python.exe quality\paint_audit.py path\to\model.glb --config path\to\profile.json
.runtime\venv\Scripts\python.exe -m unittest discover -s quality -p test_paint_audit.py
```

The same CLI accepts old and v2 GLBs, but a region is measurable only when the specified material has an embedded PNG/JPEG base color texture and uncompressed UV0. Unsupported, missing, ambiguous, transformed, transparent or extended material/texture data is `unmeasured`, as is an invalid rectangle. A solid factor with no texture is unmeasured. Review the actual JSON for each run; invocation success is not a finding of visual quality.

Rectangles are user-selected atlas areas, not computed visibility or UV-triangle masks: they can include empty atlas space, multiple parts, seams, padding and overlaps, or omit visible areas. The tool does not verify UV bounds, texture sampling/filtering, occlusion, lighting, metallic/roughness, normal maps, vertex colors, decals, geometry, view coverage, color management of renderers, or whether a flagged island is intended. Palette and thresholds require human selection; gradients and antialiasing can create findings or hide defects. A clean scan is not proof of clean paint. Profile and inspect each relevant atlas part and each view separately.

## Resource limits and malformed input

The CLI reads at most 64 KiB of profile JSON and rejects GLBs larger than 64 MiB before loading. Profiles support at most 32 regions, 16 colors per region, 250,000 pixels per rectangle and 2,000,000 total pixel × palette comparisons. Textures above 16,777,216 pixels are unmeasured. Invalid profile structure and exceeded global limits return JSON `error` with exit code 2; an unsupported material or invalid image produces a region-level `unmeasured` result. Input remains local; this tool makes no network requests. Limits are ceilings, not guarantees that a scan will be quick or that the selected area is meaningful.

## Interpreting coverage

A rectangle only measures its selected atlas pixels. Map it to the model surface before attributing a finding to a foot, face or body. Compare unlit base color and lit renders from the same GLB when reviewing bright spots or unexpected dark marks.
