---
name: setup
description: Set up the local Image to 3D Studio plugin and diagnose its service.
---
Check Node 22 and a local checkout. Run `npm ci` inside the checkout's `clients` directory. Follow `clients/README.md` to register the marketplace and run `plugins/image-to-3d-studio/bootstrap.mjs` with the explicitly chosen Codex home and checkout. Start the local studio service using its documented instructions. Restart Codex, then call the `diagnose` tool. Do not edit user-wide configuration without authorization, install globally, or expose the service on the network. Report missing prerequisites plainly.
