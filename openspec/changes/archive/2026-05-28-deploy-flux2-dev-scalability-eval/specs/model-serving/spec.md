## ADDED Requirements

### Requirement: HTTP image-editing endpoint

The server SHALL expose a `POST /edit` endpoint that accepts a multipart form request containing an image file (PNG or JPG) and a text prompt describing the desired edit. The endpoint SHALL also accept an optional `image_hash` form field (SHA-256 hex string) to enable VAE latent cache lookups. The endpoint SHALL return a JSON response containing the edited image as a base64-encoded PNG string and per-request metrics.

#### Scenario: Successful image edit

- **WHEN** a client sends a POST request to `/edit` with a valid PNG image and prompt "Change the hat color to red"
- **THEN** the server returns HTTP 200 with a JSON body containing `image_base64` (a valid base64-encoded PNG), `request_id` (a UUID string), and a `metrics` object

#### Scenario: Successful edit with JPG input

- **WHEN** a client sends a POST request to `/edit` with a valid JPG image and a prompt
- **THEN** the server accepts the JPG input and returns the edited image as a base64-encoded PNG in the same response format as for PNG input

#### Scenario: Missing image field

- **WHEN** a client sends a POST request to `/edit` without an image file
- **THEN** the server returns HTTP 422 with a validation error

#### Scenario: Missing prompt field

- **WHEN** a client sends a POST request to `/edit` without a prompt string
- **THEN** the server returns HTTP 422 with a validation error

### Requirement: VAE latent caching for related requests

The server SHALL cache VAE-encoded latent representations of source images, keyed by a SHA-256 hash of the raw image bytes. When a request arrives with an `image_hash` form field matching a cached entry on the assigned GPU, the server SHALL skip VAE encoding and reuse the cached latents. Each GPU SHALL maintain its own independent latent cache.

#### Scenario: Cold start — first request for an image

- **WHEN** a request arrives with a source image that has not been seen before on the assigned GPU
- **THEN** the server computes the VAE encoding, stores the latent tensor in that GPU's cache keyed by the image's SHA-256 hash, and proceeds with inference

#### Scenario: Warm hit — subsequent request for the same image

- **WHEN** a request arrives with `image_hash` matching an entry in the assigned GPU's latent cache
- **THEN** the server skips VAE encoding, reuses the cached latent tensor, and proceeds directly to the denoising steps

#### Scenario: Client omits image_hash

- **WHEN** a request arrives without the `image_hash` field but includes an image file
- **THEN** the server computes the SHA-256 hash from the uploaded image bytes, checks the cache, and behaves as cold-start or warm-hit accordingly

#### Scenario: Client provides image_hash without image file

- **WHEN** a request arrives with `image_hash` but no image file, and the hash exists in the assigned GPU's cache
- **THEN** the server uses the cached latents without requiring the image to be uploaded again

#### Scenario: Client provides image_hash without image file and cache miss

- **WHEN** a request arrives with `image_hash` but no image file, and the hash does NOT exist in the assigned GPU's cache
- **THEN** the server returns HTTP 404 with an error indicating the cached latent was not found and the image must be uploaded

### Requirement: Configurable inference parameters

The `/edit` endpoint SHALL accept optional form fields `num_steps` (integer, default 28) and `guidance_scale` (float, default 4.0) to control inference behavior.

#### Scenario: Custom step count

- **WHEN** a client sends a request with `num_steps=50`
- **THEN** the server runs inference with 50 denoising steps and includes `num_steps: 50` in the response metrics

#### Scenario: Default parameters

- **WHEN** a client sends a request without `num_steps` or `guidance_scale`
- **THEN** the server uses 28 steps and guidance scale 4.0

### Requirement: Model loading at startup

The server SHALL load the FLUX.2-dev diffusers pipeline onto all available CUDA GPUs during application startup, before accepting any requests. The server SHALL NOT accept requests until all model replicas are fully loaded.

#### Scenario: Startup with 4 GPUs

- **WHEN** the server starts on a machine with 4 CUDA GPUs
- **THEN** the server loads one model replica per GPU and begins accepting HTTP requests on port 8000

#### Scenario: Startup with pre-downloaded weights

- **WHEN** the `FLUX2_MODEL_PATH` environment variable is set to a local directory containing model weights
- **THEN** the server loads weights from that local path instead of downloading from HuggingFace
