"""Pipeline identifiers saved with each job, including interrupted jobs."""

SF3D_PIPELINE = "sf3d-official-preview"
PIXAL_PIPELINE = "pixal3d-space-v1"
PIXAL_RECIPIENT = "Hugging Face user victor / Space victor/pixal3d-studio"
PIXAL_AGREEMENT = {
    "schema": 1, "recipient": PIXAL_RECIPIENT,
    "source": "user-attested public CC0 photo only", "non_official_space": True,
    "retention": "public demo; upload and output retention not guaranteed",
    "quota": "anonymous ZeroGPU only; no payment or account credentials",
    "max_generation_requests": 1,
}
SF3D_AGREEMENT = {
    "recipient": "Stability AI / Hugging Face Space stabilityai/stable-fast-3d",
    "image": "locally selected and segmented photograph", "quota": "personal anonymous ZeroGPU allowance",
    "purpose": "preview evaluation", "schema": 1,
}
LEGACY_PIPELINE = "triposr-auto-finish-v1"
PIPELINE = "triposr-auto-finish-v2"
SHAPE_PIPELINE = "triposg-local-v1"
GUIDED_PIPELINE = "triposg-codex-guided-v1"
HOST_PIPELINE = "triposg-host-guided-v1"
OBSERVED_PIPELINE = "triposg-codex-observed-v1"
SF3D_OBSERVED_PIPELINE = "sf3d-codex-observed-v1"
SF3D_PIPELINES = (SF3D_PIPELINE, SF3D_OBSERVED_PIPELINE)
SPACE_PIPELINES = (*SF3D_PIPELINES, PIXAL_PIPELINE)
OBSERVED_PIPELINES = (OBSERVED_PIPELINE, SF3D_OBSERVED_PIPELINE)
OBSERVATION_METHODS = ("codex-photo-observations-v1", "codex-photo-observations-v2")
OBSERVED_AGREEMENT = {
    "schema": 1, "provider": "OpenAI Codex CLI / ChatGPT vision and image_generation",
    "recipient": "OpenAI ChatGPT", "personal_subscription": True, "api_key_billing": False,
    "send": ["normalized source photo for text observation",
             "normalized source photo and two rendered geometry views for inferred paint"],
    "max_photo_observations_per_job": 1, "max_image_generations_per_job": 1,
    "max_submissions_per_job": 2, "billing": "personal ChatGPT plan and account credit settings",
}
GUIDED_PIPELINES = (GUIDED_PIPELINE, HOST_PIPELINE, OBSERVED_PIPELINE)
SHAPE_PIPELINES = (SHAPE_PIPELINE, *GUIDED_PIPELINES)
