---
name: production
description: Generate and inspect 3D output from an explicitly selected image using the local studio.
---
Use the local Image to 3D Studio MCP tools. Diagnose first, then submit the exact selected PNG/JPEG/WebP bytes as base64 with a stable request key; preserve the returned job ID. Use wait for SSE completion and status for detail. Cancel or resume only on request. Retrieve artifacts in bounded chunks or guide the user to the local preview/export. Inspect outputs and state that unseen geometry is inferred and results require review. Never claim completion without examining the actual result. Do not send files to remote services or assume an attachment is locally accessible.


If baseline visual review fails and the host offers image generation, first explain that a three-quarter-view figurine reference is inferred, identify the intended subject, and obtain consent before sending any image to an external service. If the user agrees and an existing entitlement covers it, try at most one auxiliary image without automatic billing. Record the original and auxiliary image hashes and provenance separately. Never claim the auxiliary image is the user's original or substitute it when no actual user reference was supplied.

Read the pipeline recorded on the job. The standard TripoSR engine can use image measurements for a specific finish. The optional TripoSG engine estimates shape directly from a single segmented photograph and paints the visible surface. With TripoSG, check generation.json, appearance.json and inspection.json, including camera ambiguity and the estimated colour on hidden surfaces. All inputs use the engine configured once for that installation. A needs_review job requires an actual visual decision on the output hash.

Inspect the parts that define this particular object. Look at the same GLB with colour and as plain clay from both sides, the rear, above, below and at an oblique angle. Check through-holes, limb or handle connections, the silhouette and the transitions in surface material. Keep an unknown or failed result visible when these checks are inconclusive. Measurements for a standing figure apply only when the picture and mesh support that profile.

For a new subject, submit its original image through the same generate tool. Check the recorded finish_profile and image_comparison in inspection.json. Look for actual handles, gaps, visible color regions, thin parts, and silhouette changes in the rendered views. For desserts, check layer transitions, berry placement and piping against the source. Treat a retained baseline or an unknown comparison as a request for visual review. Preserve the distinction between a measured reconstruction, an inferred back, and an approved example.
