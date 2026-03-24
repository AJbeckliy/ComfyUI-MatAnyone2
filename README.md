# ComfyUI-MatAnyone2

ComfyUI custom nodes for MatAnyone2 video matting.

This plugin is focused on a practical end-to-end workflow inside ComfyUI:

- load a video
- choose the mask anchor frame
- preview and export that frame for mask creation
- run MatAnyone2 video matting
- save the result as MP4, transparent WebM, or transparent MOV
- preserve original audio when `audio_source` is connected

## Features

- Dedicated `MatAnyone2` node category
- `Load Video For MatAnyone2` with video upload, frame count, FPS, audio path, and selected mask-frame preview
- `MatAnyone2 Video Matting` with `mask_frame_index` support
- `Save MatAnyone2 Video` for standard MP4 output
- `Save MatAnyone2 Transparent WebM` for transparent VP9 WebM output
- `Save MatAnyone2 Transparent MOV` for transparent ProRes 4444 MOV output
- Built-in node video preview for saved outputs
- Models stored under `ComfyUI/models/matanyone2` instead of the system cache directory

## Installation

Clone into your ComfyUI `custom_nodes` folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/<your-account>/ComfyUI-MatAnyone2.git
```

Install dependencies in the same Python environment used by ComfyUI:

```bash
cd ComfyUI-MatAnyone2
pip install -r requirements.txt
```

Restart ComfyUI after installation.

## Model Location

The plugin will automatically use or download model files into:

```text
ComfyUI/models/matanyone2/MatAnyone2
ComfyUI/models/matanyone2/torch_hub/checkpoints
```

Manual model download:

- Quark: [https://pan.quark.cn/s/c3d88e3f9ece](https://pan.quark.cn/s/c3d88e3f9ece)

If you download the model manually, place the files here:

```text
ComfyUI/models/matanyone2/MatAnyone2
```

If the package also includes ResNet checkpoint files such as:

- `resnet18-5c106cde.pth`
- `resnet50-19c8e357.pth`

place them here:

```text
ComfyUI/models/matanyone2/torch_hub/checkpoints
```

## Nodes

### Load Video For MatAnyone2

Outputs:

- `frames`
- `mask_frame_image`
- `frame_count`
- `mask_frame_index`
- `fps`
- `audio_source`

`mask_frame` controls which frame is exported as `mask_frame_image` and which index is passed downstream.

### MatAnyone2 Video Matting

Inputs:

- `images`
- `mask_frame_index`
- `invert_mask`
- `mask`

Outputs:

- `rgba_image`
- `alpha_mask`

### Save MatAnyone2 Video

Standard MP4 output with background compositing and optional alpha-video export.

### Save MatAnyone2 Transparent WebM

Transparent VP9 WebM output.

### Save MatAnyone2 Transparent MOV

Transparent ProRes 4444 MOV output for editing tools. The node preview uses a browser-friendly proxy video while the real saved file remains transparent MOV.

## Example Workflows

- `example_video_workflow_full_pipeline.json`
- `example_video_workflow_transparent_webm.json`
- `example_video_workflow_transparent_mov.json`

Each workflow includes:

- video input
- selected mask-frame preview/save
- MatAnyone2 matting
- final video export

## Notes

- MatAnyone2 is a video matting model, not a generic one-click single-image background remover.
- You still need a foreground mask for the selected `mask_frame`.
- If you use a separate black/white mask image, keep `invert_mask = false`.
- If you use a ComfyUI alpha-derived mask and the result looks inverted, try `invert_mask = true`.

## Credits

- [MatAnyone2](https://github.com/pq-yang/MatAnyone2)
- Bilibili: [https://space.bilibili.com/481121504?spm_id_from=333.1007.0.0](https://space.bilibili.com/481121504?spm_id_from=333.1007.0.0)
