# Continuous surface casting

`cast_surface.py` is an optional, local geometry finishing helper. It takes explicit measurements chosen by the operator and combines ellipsoids and tapered round tubes into a continuous scalar field, applies a smooth minimum at joins, then extracts the zero surface with marching cubes. It does not segment an image, infer the requested parts, fit them to a model, transfer materials, or guarantee a printable result. Curves are specified as polylines; add control points to approximate a curve. Continuous distances avoid the concentric voxel-distance bands that can result from smoothing a binary occupancy grid, although mesh resolution and faceting still depend on pitch.

```json
{
  "pitch": 0.04,
  "blend": 0.08,
  "primitives": [
    {"type":"ellipsoid", "center":[0,0,0], "radii":[0.5,0.5,0.5]},
    {"type":"tube", "points":[[0.3,0,0],[0.7,0.1,0],[0.9,0.3,0]], "radii":[0.18,0.18,0.14]}
  ]
}
```

Run from the repository root:

```powershell
.\.runtime\venv\Scripts\python.exe engine\cast_surface.py recipe.json output.glb
```

Output can be `.glb` or `.ply`. Coordinates and units are those in the recipe; no automatic orientation or scale is applied. The output is a neutral mesh with no texture or preserved names. The output path must not already exist. No network access, code execution, external file references, or asset fetching is part of the recipe.

All values must be finite numbers; bools and unrecognized fields are rejected. `pitch` is in [0.0001, 100], `blend` in [0, 100], and each radius is at least twice pitch. Coordinates and individual numerical magnitudes cannot exceed 10,000. A tube needs at least two distinct consecutive points and one radius per point. At most 64 primitives, 128 total segments/ellipsoids, 4,000,000 grid samples and 250,000,000 sample-segment evaluations are allowed. The input JSON is limited to 128 KiB. Field evaluation uses chunks of 100,000 samples; marching cubes and the final mesh add memory beyond the 16 MB maximum field array. On rejection, increase pitch or reduce part count/extent. The surface is rejected if it touches the sampling box boundary. Narrow gaps can close when the blend width is large; inspect the output and choose blend to preserve intended gaps.

To check the generic examples and input limits:

```powershell
.\.runtime\venv\Scripts\python.exe -m unittest quality.test_cast_surface -v
```
