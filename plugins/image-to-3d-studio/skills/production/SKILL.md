---
name: production
description: Generate and inspect 3D output from an explicitly selected image using the local studio.
---
Use the local Image to 3D Studio MCP tools. Diagnose first, then submit the exact selected PNG/JPEG/WebP bytes as base64 with a stable request key; preserve the returned job ID. Use wait for SSE completion and status for detail. Cancel or resume only on request. Retrieve artifacts in bounded chunks or guide the user to the local preview/export. Inspect outputs and state that unseen geometry is inferred and results require review. Never claim completion without examining the actual result. Do not send files to remote services or assume an attachment is locally accessible.


If baseline visual review fails and the host offers image generation, first explain that a three-quarter-view figurine reference is inferred, identify the intended subject, and obtain consent before sending any image to an external service. If the user agrees and an existing entitlement covers it, try at most one auxiliary image without automatic billing. Record the original and auxiliary image hashes and provenance separately. Never claim the auxiliary image is the user's original or substitute it when no actual user reference was supplied.
