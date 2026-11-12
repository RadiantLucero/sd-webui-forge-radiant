# Codebase architecture

This is a navigation map for the current implementation, not a complete API reference. Follow the code when behavior and this overview differ.

## Runtime startup

1. [`launch.py`](../launch.py) delegates environment setup and command-line handling to [`modules/launch_utils.py`](../modules/launch_utils.py).
2. `launch_utils.start()` imports [`webui.py`](../webui.py). Importing that module initializes Forge and the shared WebUI state before it starts either the Gradio UI or API-only mode.
3. [`modules_forge/initialization.py`](../modules_forge/initialization.py) configures the Forge backend, device and memory behavior, and related patches. [`modules/initialize.py`](../modules/initialize.py) initializes shared WebUI state, model metadata, extensions, scripts, samplers, and related UI data.
4. In UI mode, `webui_worker()` runs pre-UI callbacks, builds the Gradio interface, attaches middleware and APIs, then runs app-started callbacks. API-only mode follows a separate startup path in `webui.py`.

`webui.py` performs significant initialization at import time. Avoid importing it from tests or utilities just to access a helper.

## Image-generation request path

The main UI is assembled in [`modules/ui.py`](../modules/ui.py). Its Gradio events call [`modules/txt2img.py`](../modules/txt2img.py) and [`modules/img2img.py`](../modules/img2img.py). The API handlers live in [`modules/api/api.py`](../modules/api/api.py).

Both UI and API paths create a `StableDiffusionProcessing` variant and ultimately use [`modules/processing.py`](../modules/processing.py). For the UI, `txt2img` and `img2img` submit the work through [`modules_forge/main_thread.py`](../modules_forge/main_thread.py), whose worker serializes the primary generation requests. Script hooks can run before the standard `process_images()` path.

For worker tasks that fail with an out-of-memory error, the worker logs the original traceback and unloads models after leaving the exception handler, before publishing completion or starting another task. This recovery is tied to the failing task, not to a later Gradio callback. An unload failure is reported without terminating the worker. This does not serialize independent model operations performed by extensions or other endpoints.

`process_images()` applies request-level option overrides, ensures the required model is loaded, and enters `process_images_inner()`. The inner loop prepares prompts and conditioning, invokes the processing object's sampler, decodes results, runs post-processing hooks, and saves or returns outputs.

Conditioning is cached in a single slot shared by txt2img and img2img. Edit-capable engines with reference conditioning enabled (for example Klein or Krea2) keep the encoded img2img input as a pending reference (`ini_latent`) that the next positive prompt encode consumes and publishes as `dynamic_args.ref_latents`. A cache hit skips that encode, so `process_images()` discards a leftover pending reference when the generation ends; otherwise a later txt2img run could adopt the img2img image.

## Model and backend responsibilities

- [`modules/sd_models.py`](../modules/sd_models.py) discovers checkpoints, tracks model state, and coordinates model reloads.
- [`backend/loader.py`](../backend/loader.py) detects and loads model components. Model-family implementations are under [`backend/diffusion_engine/`](../backend/diffusion_engine/).
- [`backend/memory_management.py`](../backend/memory_management.py), `backend/patcher/`, `backend/nn/`, and `backend/sampling/` contain lower-level device, model-patching, network, and sampling code.
- [`modules_forge/`](../modules_forge/) connects the WebUI to Forge-specific controls and backend behavior. `modules_forge/packages/` contains bundled packages; inspect their provenance and license before changing them.

Model loading and sampling interact with GPU memory and model-specific behavior. A change in one layer can affect switching checkpoints, refiner use, or subsequent generations, so trace those paths before changing lifecycle or cache behavior.

## Extensions and scripts

- [`extensions-builtin/`](../extensions-builtin/) contains built-in extensions. User-installed extensions are loaded from the runtime extensions directory and are not part of this repository's tracked source.
- [`modules/extensions.py`](../modules/extensions.py) discovers extensions and reads their metadata and enablement requirements.
- [`modules/scripts.py`](../modules/scripts.py) loads scripts from the root `scripts/` directory and active extensions. Script runners expose lifecycle hooks around processing and post-processing.
- [`modules/script_callbacks.py`](../modules/script_callbacks.py) provides callbacks for application, model, UI, and image lifecycle events.

An extension can register UI components and alter generation behavior through scripts or callbacks. When debugging a behavior change, check both the core path and the relevant extension hooks.

## Browser UI

The Gradio interface is primarily assembled in `modules/ui.py`. Browser-side behavior and styling are in [`javascript/`](../javascript/), [`html/`](../html/), [`script.js`](../script.js), and [`style.css`](../style.css). Many UI event bindings are created in Python and depend on component IDs used by the JavaScript and CSS; keep those sides consistent.

## Practical entry points

| Change area | Start here |
| --- | --- |
| Startup, command-line options, initialization | `launch.py`, `modules/launch_utils.py`, `webui.py`, `modules/initialize.py` |
| txt2img / img2img behavior | `modules/txt2img.py`, `modules/img2img.py`, `modules/processing.py` |
| API generation behavior | `modules/api/api.py`, then the corresponding processing path |
| Checkpoint discovery or switching | `modules/sd_models.py`, `backend/loader.py` |
| GPU placement or memory use | `backend/memory_management.py`, `backend/patcher/`, `modules_forge/main_thread.py` |
| Extension or script behavior | `modules/extensions.py`, `modules/scripts.py`, `modules/script_callbacks.py`, the extension itself |
| UI layout or browser interaction | `modules/ui.py`, then the related JavaScript, HTML, and CSS |

## Validation constraints

The generation runtime depends on platform-specific PyTorch and GPU support. There is no project-level automated test suite at present. Use Ruff for changed Python files, add focused tests when behavior can be isolated from the runtime, and report GPU-dependent checks that could not be run.
