# PiD decoder (`--pid-decode`) test images

Comparison crops supporting my review of filipstrand/mflux#490.

Each image is a 1:1 side-by-side at native resolution:
**left** = VAE decode of the same latent, Lanczos-upscaled 4x;
**right** = PiD decode. Same prompt, same seed, same generation size per pair.

- `portrait-*` — 512x640 generated, PiD output 2048x2560
- `puffin-*` — 640x512 generated, PiD output 2560x2048

Crop boxes differ per model because each frames the subject differently;
within a pair both sides use the identical box.
