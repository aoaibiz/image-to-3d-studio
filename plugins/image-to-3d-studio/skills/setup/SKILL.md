---
name: setup
description: Set up the local Image to 3D Studio plugin and diagnose its service.
---
Check Node 22 and a local checkout. Run `npm ci` inside the checkout's `clients` directory. Follow `clients/README.md` to register the marketplace and run `plugins/image-to-3d-studio/bootstrap.mjs` with the explicitly chosen Codex home and checkout. Start the local studio service using its documented instructions. Restart Codex, then call the `diagnose` tool. Do not edit user-wide configuration without authorization, install globally, or expose the service on the network. Report missing prerequisites plainly.

The standard local engine is TripoSR. If the operator chooses the optional TripoSG shape engine during setup, use engine/TRIPOSG.md in the selected checkout, configure the job store once, and restart that Studio service. Explain the model download size, CPU time and upstream territorial terms. Check the selected pipeline with the diagnose tool. The image stays on the user's computer with either engine.

For an operator who chooses native Codex ImageGen for back-side paint, explain that a metadata-stripped photo and two-view geometry guide will go to OpenAI ChatGPT through their host account, up to one call per job, subject to that account's usage limits. On consent, configure `python -m studio.configure --engine triposg-guided-host --allow-send-to-openai`, then restart Studio. This host and Studio must share a filesystem. The CLI/MCP reads the generated local token from the Studio job store. For a custom store, set `STUDIO_HOST_TOKEN_FILE` to its `host-access.token` path. Keep its contents private. Return to `--engine triposg` and restart Studio to use local paint again.
