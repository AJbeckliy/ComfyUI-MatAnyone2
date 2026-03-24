import os
import shutil
import subprocess
import time

import cv2
import folder_paths
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from huggingface_hub import snapshot_download


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".mpg",
    ".mpeg",
    ".m4v",
}

MATANYONE2_CATEGORY = "MatAnyone2"

MATANYONE2_REPO_ID = "PeiqingYang/MatAnyone2"
MATANYONE2_MODELS_DIR = os.path.join(folder_paths.models_dir, "matanyone2")
MATANYONE2_LOCAL_DIR = os.path.join(MATANYONE2_MODELS_DIR, "MatAnyone2")
MATANYONE2_TORCH_HOME = os.path.join(MATANYONE2_MODELS_DIR, "torch_hub")
LEGACY_HF_CACHE_DIR = os.path.join(
    os.path.expanduser("~"),
    ".cache",
    "huggingface",
    "hub",
    "models--PeiqingYang--MatAnyone2",
    "snapshots",
)
LEGACY_TORCH_CHECKPOINT_DIR = os.path.join(
    os.path.expanduser("~"),
    ".cache",
    "torch",
    "hub",
    "checkpoints",
)
RESNET_CHECKPOINTS = [
    "resnet18-5c106cde.pth",
    "resnet50-19c8e357.pth",
]


def list_input_videos():
    input_dir = folder_paths.get_input_directory()
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    files = folder_paths.filter_files_content_types(files, ["video"])
    files = [f for f in files if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS]
    return sorted(files)


def normalize_relative_save_path(save_path):
    save_path = (save_path or "").strip().replace("\\", "/").strip("/")
    if not save_path:
        return ""

    normalized = os.path.normpath(save_path).replace("\\", "/")
    if normalized.startswith("..") or normalized == "." or os.path.isabs(normalized):
        raise RuntimeError("save_path must be a relative path inside the ComfyUI output directory, or an absolute directory.")
    return normalized.strip("/")


def build_filename_prefix(filename_prefix, save_path):
    prefix = (filename_prefix or "").strip().replace("\\", "/").strip("/")
    subfolder = normalize_relative_save_path(save_path)

    if subfolder and prefix:
        return f"{subfolder}/{prefix}"
    if subfolder:
        return subfolder
    return prefix


def compute_prefix_vars(value, image_width, image_height):
    value = value or ""
    value = value.replace("%width%", str(image_width))
    value = value.replace("%height%", str(image_height))
    now = time.localtime()
    value = value.replace("%year%", str(now.tm_year))
    value = value.replace("%month%", str(now.tm_mon).zfill(2))
    value = value.replace("%day%", str(now.tm_mday).zfill(2))
    value = value.replace("%hour%", str(now.tm_hour).zfill(2))
    value = value.replace("%minute%", str(now.tm_min).zfill(2))
    value = value.replace("%second%", str(now.tm_sec).zfill(2))
    return value


def next_counter(full_output_folder, filename):
    prefix_len = len(filename)
    try:
        matches = []
        for entry in os.listdir(full_output_folder):
            prefix = entry[: prefix_len + 1]
            if os.path.normcase(prefix[:-1]) != os.path.normcase(filename):
                continue
            if prefix[-1] != "_":
                continue
            digits = int(entry[prefix_len + 1 :].split("_")[0])
            matches.append(digits)
        return (max(matches) + 1) if matches else 1
    except FileNotFoundError:
        os.makedirs(full_output_folder, exist_ok=True)
        return 1


def resolve_output_location(filename_prefix, save_path, output_dir, image_width, image_height):
    raw_prefix = (filename_prefix or "").strip().replace("\\", "/").strip("/")
    raw_save_path = (save_path or "").strip()

    if raw_save_path and os.path.isabs(os.path.normpath(raw_save_path)):
        resolved_prefix = compute_prefix_vars(raw_prefix or "output", image_width, image_height)
        normalized_prefix = os.path.normpath(resolved_prefix)
        filename = os.path.basename(normalized_prefix) or "output"
        prefix_subfolder = os.path.dirname(normalized_prefix)
        full_output_folder = os.path.normpath(
            os.path.join(raw_save_path, prefix_subfolder) if prefix_subfolder else raw_save_path
        )
        os.makedirs(full_output_folder, exist_ok=True)
        counter = next_counter(full_output_folder, filename)
        return full_output_folder, filename, counter, "", True

    effective_prefix = build_filename_prefix(raw_prefix, raw_save_path)
    full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        effective_prefix, output_dir, image_width, image_height
    )
    return full_output_folder, filename, counter, subfolder, False


def write_preview_png(full_output_folder, filename, counter, preview_rgb):
    os.makedirs(full_output_folder, exist_ok=True)
    preview_file = f"{filename}_{counter:05}_preview.png"
    preview_path = os.path.join(full_output_folder, preview_file)
    Image.fromarray(preview_rgb).save(preview_path)
    return preview_file


def ensure_ui_video_preview(video_path, output_dir, ui_subfolder="MatAnyone2/previews"):
    if not video_path or not os.path.isfile(video_path):
        return []

    abs_output_dir = os.path.abspath(output_dir)
    abs_video_path = os.path.abspath(video_path)
    video_name = os.path.basename(video_path)
    video_ext = os.path.splitext(video_name)[1].lower()
    if video_ext == ".webm":
        mime_type = "video/webm"
    elif video_ext == ".mov":
        mime_type = "video/quicktime"
    else:
        mime_type = "video/mp4"

    try:
        common = os.path.commonpath((abs_output_dir, abs_video_path))
    except ValueError:
        common = None

    if common == abs_output_dir:
        rel_subfolder = os.path.relpath(os.path.dirname(abs_video_path), abs_output_dir)
        rel_subfolder = "" if rel_subfolder == "." else rel_subfolder.replace("\\", "/")
        return [{
            "filename": video_name,
            "subfolder": rel_subfolder,
            "type": "output",
            "format": mime_type,
        }]

    ui_folder = os.path.join(output_dir, ui_subfolder)
    os.makedirs(ui_folder, exist_ok=True)
    ui_target = os.path.join(ui_folder, video_name)
    if os.path.normcase(os.path.abspath(ui_target)) != os.path.normcase(abs_video_path):
        shutil.copy2(abs_video_path, ui_target)
    return [{
        "filename": video_name,
        "subfolder": ui_subfolder.replace("\\", "/"),
        "type": "output",
        "format": mime_type,
    }]


def write_ui_proxy_preview_video(output_dir, filename_prefix, images, alpha_mask, fps, audio_source=None):
    import imageio_ffmpeg

    if images.ndim != 4 or images.shape[-1] < 3:
        raise RuntimeError(f"Expected IMAGE input with shape (B, H, W, C), got {tuple(images.shape)}")

    image_height = int(images.shape[1])
    image_width = int(images.shape[2])

    ui_prefix = build_filename_prefix(filename_prefix, "MatAnyone2/previews")
    ui_folder, ui_filename, ui_counter, _, _ = folder_paths.get_save_image_path(
        ui_prefix, output_dir, image_width, image_height
    )
    proxy_file = f"{ui_filename}_{ui_counter:05}_.mp4"
    proxy_path = os.path.join(ui_folder, proxy_file)

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    audio_input_args = []
    audio_output_args = ["-an"]
    if audio_source:
        audio_source = audio_source.strip()
        if audio_source:
            if not os.path.exists(audio_source):
                raise RuntimeError(f"Audio source does not exist: {audio_source}")
            audio_input_args = ["-i", audio_source]
            audio_output_args = ["-map", "1:a?", "-c:a", "aac", "-shortest"]

    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-nostats",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{image_width}x{image_height}",
        "-r",
        str(float(fps)),
        "-i",
        "-",
        *audio_input_args,
        "-map",
        "0:v:0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        *audio_output_args,
        proxy_path,
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        for frame_idx in range(images.shape[0]):
            rgb = np.clip(images[frame_idx, ..., :3].float().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
            if alpha_mask is not None:
                alpha = alpha_mask[frame_idx].float().cpu().numpy()
            elif images.shape[-1] >= 4:
                alpha = images[frame_idx, ..., 3].float().cpu().numpy()
            else:
                alpha = np.ones((image_height, image_width), dtype=np.float32)
            alpha = np.clip(alpha, 0.0, 1.0)[..., None]
            composite = np.clip(rgb.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
            proc.stdin.write(composite.tobytes())
        proc.stdin.close()
        stderr = proc.stderr.read()
        returncode = proc.wait()
    except Exception:
        proc.kill()
        proc.wait()
        raise
    if returncode != 0:
        raise RuntimeError(f"Failed to save preview MP4 video:\n{stderr.decode('utf-8', 'ignore')}")

    return proxy_path


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def copy_if_missing(src, dst):
    if os.path.exists(dst) or not os.path.exists(src):
        return
    ensure_dir(os.path.dirname(dst))
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)


def ensure_matanyone2_model_dir():
    ensure_dir(MATANYONE2_LOCAL_DIR)

    config_path = os.path.join(MATANYONE2_LOCAL_DIR, "config.json")
    weight_candidates = [
        os.path.join(MATANYONE2_LOCAL_DIR, "model.safetensors"),
        os.path.join(MATANYONE2_LOCAL_DIR, "pytorch_model.bin"),
    ]
    has_local_model = os.path.exists(config_path) and any(os.path.exists(p) for p in weight_candidates)
    if has_local_model:
        return MATANYONE2_LOCAL_DIR

    if os.path.isdir(LEGACY_HF_CACHE_DIR):
        snapshots = [
            os.path.join(LEGACY_HF_CACHE_DIR, name)
            for name in os.listdir(LEGACY_HF_CACHE_DIR)
            if os.path.isdir(os.path.join(LEGACY_HF_CACHE_DIR, name))
        ]
        if snapshots:
            latest_snapshot = max(snapshots, key=os.path.getmtime)
            for name in os.listdir(latest_snapshot):
                copy_if_missing(
                    os.path.join(latest_snapshot, name),
                    os.path.join(MATANYONE2_LOCAL_DIR, name),
                )

    has_local_model = os.path.exists(config_path) and any(os.path.exists(p) for p in weight_candidates)
    if not has_local_model:
        snapshot_download(
            repo_id=MATANYONE2_REPO_ID,
            local_dir=MATANYONE2_LOCAL_DIR,
            local_dir_use_symlinks=False,
        )

    return MATANYONE2_LOCAL_DIR


def ensure_matanyone2_torch_home():
    checkpoints_dir = ensure_dir(os.path.join(MATANYONE2_TORCH_HOME, "checkpoints"))
    for filename in RESNET_CHECKPOINTS:
        copy_if_missing(
            os.path.join(LEGACY_TORCH_CHECKPOINT_DIR, filename),
            os.path.join(checkpoints_dir, filename),
        )
    os.environ["TORCH_HOME"] = MATANYONE2_TORCH_HOME
    torch.hub.set_dir(MATANYONE2_TORCH_HOME)
    return MATANYONE2_TORCH_HOME


ensure_dir(MATANYONE2_MODELS_DIR)


class LoadVideoForMatAnyone2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (list_input_videos(), {"video_upload": True, "image_upload": True}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "mask_frame": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("frames", "mask_frame_image", "frame_count", "mask_frame_index", "fps", "audio_source")
    FUNCTION = "load_video"
    CATEGORY = MATANYONE2_CATEGORY

    @classmethod
    def IS_CHANGED(cls, video, max_frames, mask_frame):
        video_path = folder_paths.get_annotated_filepath(video)
        stat = os.stat(video_path)
        return f"{stat.st_mtime_ns}:{stat.st_size}:{max_frames}:{mask_frame}"

    @classmethod
    def VALIDATE_INPUTS(cls, video, max_frames, mask_frame):
        if not folder_paths.exists_annotated_filepath(video):
            return f"Invalid video file: {video}"
        return True

    def load_video(self, video, max_frames, mask_frame):
        video_path = folder_paths.get_annotated_filepath(video)
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = []

        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb.astype(np.float32) / 255.0)
            frames.append(frame_tensor)

            if max_frames > 0 and len(frames) >= max_frames:
                break

        capture.release()

        if not frames:
            raise RuntimeError(f"No frames found in video: {video_path}")

        frames_tensor = torch.stack(frames, dim=0)
        frame_count = int(frames_tensor.shape[0])
        if mask_frame < 0 or mask_frame >= frame_count:
            raise RuntimeError(f"mask_frame {mask_frame} is out of range for loaded video with {frame_count} frames.")
        mask_frame_image = frames_tensor[mask_frame : mask_frame + 1]

        return (frames_tensor, mask_frame_image, frame_count, int(mask_frame), fps, video_path)


class MatAnyone2Node:
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "mask_frame_index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1, "forceInput": True}),
                "invert_mask": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("rgba_image", "alpha_mask")
    FUNCTION = "process"
    CATEGORY = MATANYONE2_CATEGORY

    def load_model(self):
        if self.model is None:
            try:
                from matanyone2 import MatAnyone2

                local_model_dir = ensure_matanyone2_model_dir()
                ensure_matanyone2_torch_home()
                self.model = MatAnyone2.from_pretrained(local_model_dir)
                self.model.to(self.device)
                self.model.eval()
            except Exception as e:
                raise RuntimeError(f"Failed to load MatAnyone2: {e}") from e
        return self.model

    def _prepare_mask(self, mask, height, width, mask_frame=0, invert_mask=False):
        if mask is None:
            raise RuntimeError(
                "MatAnyone2 requires an initial MASK for the first frame. "
                "Connect a subject mask to the 'mask' input."
            )

        if mask.ndim == 2:
            first_mask = mask
        elif mask.ndim == 3:
            if mask.shape[0] == 0:
                raise RuntimeError("Received an empty MASK sequence.")
            if mask.shape[0] == 1:
                first_mask = mask[0]
            else:
                if mask_frame < 0 or mask_frame >= mask.shape[0]:
                    raise RuntimeError(
                        f"mask_frame {mask_frame} is out of range for MASK input with {mask.shape[0]} frames."
                    )
                first_mask = mask[mask_frame]
        else:
            raise RuntimeError(f"Unsupported mask shape: {tuple(mask.shape)}")

        first_mask = first_mask.float().to(self.device)

        if first_mask.shape[-2:] != (height, width):
            first_mask = F.interpolate(
                first_mask.unsqueeze(0).unsqueeze(0),
                size=(height, width),
                mode="nearest",
            )[0, 0]

        if first_mask.max().item() > 1.0:
            first_mask = first_mask / 255.0

        first_mask = first_mask.clamp(0.0, 1.0)
        if invert_mask:
            first_mask = 1.0 - first_mask

        first_mask = first_mask * 255.0

        return first_mask.clamp(0.0, 255.0)

    def _run_sequence(self, frames, frame_indices, init_mask, device, warmup_steps=10):
        from matanyone2.inference.inference_core import InferenceCore
        from matanyone2.utils.device import safe_autocast

        processor = InferenceCore(self.model, cfg=self.model.cfg, device=device)
        alpha_frames = []
        frame_indices = list(frame_indices)
        if not frame_indices:
            raise RuntimeError("Cannot process an empty frame sequence.")

        with torch.inference_mode():
            with safe_autocast():
                first_frame = frames[frame_indices[0]].to(device)
                processor.step(first_frame, init_mask, objects=[1])

                output_prob = None
                for _ in range(max(int(warmup_steps), 0)):
                    output_prob = processor.step(first_frame, first_frame_pred=True)

                if output_prob is None:
                    output_prob = processor.step(first_frame, first_frame_pred=True)

                alpha_frames.append(processor.output_prob_to_mask(output_prob).float().cpu())
                del first_frame, output_prob

                for frame_idx in frame_indices[1:]:
                    frame = frames[frame_idx].to(device)
                    output_prob = processor.step(frame)
                    alpha_frames.append(processor.output_prob_to_mask(output_prob).float().cpu())
                    del frame, output_prob

        del processor
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return torch.stack(alpha_frames, dim=0).float().clamp(0.0, 1.0)

    def process(self, images, mask_frame_index=0, invert_mask=False, mask=None):
        model = self.load_model()
        self.model = model

        if images.ndim != 4 or images.shape[-1] < 3:
            raise RuntimeError(
                f"Expected IMAGE input with shape (B, H, W, C), got {tuple(images.shape)}"
            )

        mask_frame = int(mask_frame_index)
        batch, height, width, _ = images.shape
        if mask_frame < 0 or mask_frame >= batch:
            raise RuntimeError(f"mask_frame {mask_frame} is out of range for IMAGE input with {batch} frames.")

        frames = images[..., :3].permute(0, 3, 1, 2).float().cpu()
        init_mask = self._prepare_mask(mask, height, width, mask_frame=mask_frame, invert_mask=invert_mask)

        forward_alpha = self._run_sequence(
            frames,
            range(mask_frame, batch),
            init_mask,
            self.device,
            warmup_steps=10,
        )
        if mask_frame == 0:
            alpha_mask = forward_alpha
        else:
            backward_alpha = self._run_sequence(
                frames,
                range(mask_frame, -1, -1),
                init_mask,
                self.device,
                warmup_steps=10,
            )
            leading_alpha = torch.flip(backward_alpha[1:], dims=[0])
            alpha_mask = torch.cat([leading_alpha, forward_alpha], dim=0)

        alpha_mask = alpha_mask.cpu()
        rgba = torch.cat([images[..., :3].cpu(), alpha_mask.unsqueeze(-1)], dim=-1)

        return (rgba, alpha_mask)


class SaveMatAnyone2Base:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    def _resolve_alpha(self, images, alpha_mask):
        if alpha_mask is not None:
            alpha = alpha_mask.float().cpu()
        elif images.shape[-1] >= 4:
            alpha = images[..., 3].float().cpu()
        else:
            alpha = torch.ones(images.shape[:3], dtype=torch.float32)
        return alpha.clamp(0.0, 1.0)

    def _get_frame_rgb_uint8(self, images, frame_idx):
        return np.clip(images[frame_idx, ..., :3].float().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)

    def _get_frame_alpha_float(self, images, alpha_mask, frame_idx):
        if alpha_mask is not None:
            alpha = alpha_mask[frame_idx].float().cpu().numpy()
        elif images.shape[-1] >= 4:
            alpha = images[frame_idx, ..., 3].float().cpu().numpy()
        else:
            alpha = np.ones(images.shape[1:3], dtype=np.float32)
        return np.clip(alpha, 0.0, 1.0)

    def _compose_frame(self, rgb_uint8, alpha_float, background_rgb):
        alpha = alpha_float[..., None]
        background = background_rgb.reshape(1, 1, 3).astype(np.float32)
        preview = (rgb_uint8.astype(np.float32) * alpha) + (background * (1.0 - alpha))
        return np.clip(preview, 0, 255).astype(np.uint8)

    def _run_ffmpeg_stream(self, cmd, frame_iter, error_prefix):
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            for frame in frame_iter:
                proc.stdin.write(frame.tobytes())
            proc.stdin.close()
            stderr = proc.stderr.read()
            returncode = proc.wait()
        except Exception:
            proc.kill()
            proc.wait()
            raise
        if returncode != 0:
            raise RuntimeError(f"{error_prefix}\n{stderr.decode('utf-8', 'ignore')}")

    def _run_ffmpeg(self, cmd, payload, error_prefix):
        proc = subprocess.run(
            cmd,
            input=payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"{error_prefix}\n{proc.stderr.decode('utf-8', 'ignore')}")

    def _build_audio_args(self, audio_source, audio_codec):
        if not audio_source:
            return [], ["-an"]

        audio_source = audio_source.strip()
        if not audio_source:
            return [], ["-an"]
        if not os.path.exists(audio_source):
            raise RuntimeError(f"Audio source does not exist: {audio_source}")

        return ["-i", audio_source], ["-map", "1:a?", "-c:a", audio_codec, "-shortest"]


class SaveMatAnyone2Video(SaveMatAnyone2Base):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("FLOAT", {"default": 24.0, "min": 0.1, "max": 240.0, "step": 0.1, "forceInput": True}),
                "filename_prefix": (
                    "STRING",
                    {"default": "video", "tooltip": "Output filename prefix without extension."},
                ),
                "save_path": (
                    "STRING",
                    {"default": "MatAnyone2", "tooltip": "Relative folder in ComfyUI/output, or an absolute directory."},
                ),
                "background": (["black", "white", "green"], {"default": "black"}),
                "save_preview_image": ("BOOLEAN", {"default": True}),
                "save_alpha_video": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "alpha_mask": ("MASK",),
                "audio_source": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_video"
    OUTPUT_NODE = True
    CATEGORY = MATANYONE2_CATEGORY

    def _get_background_color(self, background):
        if background == "white":
            return np.array([255.0, 255.0, 255.0], dtype=np.float32)
        if background == "green":
            return np.array([0.0, 255.0, 0.0], dtype=np.float32)
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def save_video(
        self,
        images,
        fps,
        filename_prefix,
        save_path,
        background,
        save_preview_image,
        save_alpha_video,
        alpha_mask=None,
        audio_source=None,
    ):
        import imageio_ffmpeg

        if images.ndim != 4 or images.shape[-1] < 3:
            raise RuntimeError(
                f"Expected IMAGE input with shape (B, H, W, C), got {tuple(images.shape)}"
            )

        image_width = int(images.shape[2])
        image_height = int(images.shape[1])
        full_output_folder, filename, counter, subfolder, external_output = resolve_output_location(
            filename_prefix, save_path, self.output_dir, image_width, image_height
        )

        alpha_source = alpha_mask if alpha_mask is not None else None
        background_color = self._get_background_color(background)

        base_name = f"{filename}_{counter:05}_"
        video_file = f"{base_name}.mp4"
        video_path = os.path.join(full_output_folder, video_file)
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

        audio_input_args, audio_output_args = self._build_audio_args(audio_source, "aac")
        cmd = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-nostats",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{image_width}x{image_height}",
            "-r",
            str(float(fps)),
            "-i",
            "-",
            *audio_input_args,
            "-map",
            "0:v:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            *audio_output_args,
            video_path,
        ]
        self._run_ffmpeg_stream(
            cmd,
            (
                self._compose_frame(
                    self._get_frame_rgb_uint8(images, frame_idx),
                    self._get_frame_alpha_float(images, alpha_source, frame_idx),
                    background_color,
                )
                for frame_idx in range(images.shape[0])
            ),
            "Failed to save MP4 video:",
        )

        results = []
        video_previews = ensure_ui_video_preview(video_path, self.output_dir)
        if save_preview_image:
            preview_rgb = self._compose_frame(
                self._get_frame_rgb_uint8(images, 0),
                self._get_frame_alpha_float(images, alpha_source, 0),
                background_color,
            )
            preview_file = write_preview_png(full_output_folder, filename, counter, preview_rgb)
            if external_output:
                ui_prefix = build_filename_prefix(f"{filename}_preview_cache", "MatAnyone2/previews")
                ui_folder, ui_filename, ui_counter, ui_subfolder, _ = folder_paths.get_save_image_path(
                    ui_prefix, self.output_dir, image_width, image_height
                )
                ui_preview_file = write_preview_png(ui_folder, ui_filename, ui_counter, preview_rgb)
                results.append({"filename": ui_preview_file, "subfolder": ui_subfolder, "type": self.type})
            else:
                results.append({"filename": preview_file, "subfolder": subfolder, "type": self.type})

        if save_alpha_video:
            alpha_file = f"{base_name}_alpha.mp4"
            alpha_path = os.path.join(full_output_folder, alpha_file)
            alpha_cmd = [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-nostats",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{image_width}x{image_height}",
                "-r",
                str(float(fps)),
                "-i",
                "-",
                "-map",
                "0:v:0",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-an",
                alpha_path,
            ]
            self._run_ffmpeg_stream(
                alpha_cmd,
                (
                    np.repeat(
                        (self._get_frame_alpha_float(images, alpha_source, frame_idx)[..., None] * 255.0)
                        .clip(0, 255)
                        .astype(np.uint8),
                        3,
                        axis=-1,
                    )
                    for frame_idx in range(images.shape[0])
                ),
                "Failed to save alpha MP4 video:",
            )

        ui = {"gifs": video_previews}
        if results:
            ui["images"] = results
        return {"ui": ui}


class SaveMatAnyone2TransparentWebM(SaveMatAnyone2Base):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("FLOAT", {"default": 24.0, "min": 0.1, "max": 240.0, "step": 0.1, "forceInput": True}),
                "filename_prefix": (
                    "STRING",
                    {"default": "transparent", "tooltip": "Output filename prefix without extension."},
                ),
                "save_path": (
                    "STRING",
                    {"default": "MatAnyone2", "tooltip": "Relative folder in ComfyUI/output, or an absolute directory."},
                ),
                "quality": ("INT", {"default": 28, "min": 4, "max": 63, "step": 1}),
                "save_preview_image": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "alpha_mask": ("MASK",),
                "audio_source": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_video"
    OUTPUT_NODE = True
    CATEGORY = MATANYONE2_CATEGORY

    def save_video(
        self,
        images,
        fps,
        filename_prefix,
        save_path,
        quality,
        save_preview_image,
        alpha_mask=None,
        audio_source=None,
    ):
        import imageio_ffmpeg

        if images.ndim != 4 or images.shape[-1] < 3:
            raise RuntimeError(
                f"Expected IMAGE input with shape (B, H, W, C), got {tuple(images.shape)}"
            )

        image_width = int(images.shape[2])
        image_height = int(images.shape[1])
        full_output_folder, filename, counter, subfolder, external_output = resolve_output_location(
            filename_prefix, save_path, self.output_dir, image_width, image_height
        )

        alpha_source = alpha_mask if alpha_mask is not None else None

        output_file = f"{filename}_{counter:05}_.webm"
        output_path = os.path.join(full_output_folder, output_file)
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

        audio_input_args, audio_output_args = self._build_audio_args(audio_source, "libopus")
        cmd = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-nostats",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-s",
            f"{image_width}x{image_height}",
            "-r",
            str(float(fps)),
            "-i",
            "-",
            *audio_input_args,
            "-map",
            "0:v:0",
            "-c:v",
            "libvpx-vp9",
            "-pix_fmt",
            "yuva420p",
            "-crf",
            str(int(quality)),
            "-b:v",
            "0",
            *audio_output_args,
            output_path,
        ]
        self._run_ffmpeg_stream(
            cmd,
            (
                np.concatenate(
                    [
                        self._get_frame_rgb_uint8(images, frame_idx),
                        (
                            self._get_frame_alpha_float(images, alpha_source, frame_idx)[..., None] * 255.0
                        )
                        .clip(0, 255)
                        .astype(np.uint8),
                    ],
                    axis=-1,
                )
                for frame_idx in range(images.shape[0])
            ),
            "Failed to save transparent WebM:",
        )

        results = []
        video_previews = ensure_ui_video_preview(output_path, self.output_dir)
        if save_preview_image:
            preview_rgb = self._compose_frame(
                self._get_frame_rgb_uint8(images, 0),
                self._get_frame_alpha_float(images, alpha_source, 0),
                np.array([0.0, 0.0, 0.0], dtype=np.float32),
            )
            preview_file = write_preview_png(full_output_folder, filename, counter, preview_rgb)
            if external_output:
                ui_prefix = build_filename_prefix(f"{filename}_preview_cache", "MatAnyone2/previews")
                ui_folder, ui_filename, ui_counter, ui_subfolder, _ = folder_paths.get_save_image_path(
                    ui_prefix, self.output_dir, image_width, image_height
                )
                ui_preview_file = write_preview_png(ui_folder, ui_filename, ui_counter, preview_rgb)
                results.append({"filename": ui_preview_file, "subfolder": ui_subfolder, "type": self.type})
            else:
                results.append({"filename": preview_file, "subfolder": subfolder, "type": self.type})

        ui = {"gifs": video_previews}
        if results:
            ui["images"] = results
        return {"ui": ui}


class SaveMatAnyone2TransparentMOV(SaveMatAnyone2Base):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("FLOAT", {"default": 24.0, "min": 0.1, "max": 240.0, "step": 0.1, "forceInput": True}),
                "filename_prefix": (
                    "STRING",
                    {"default": "transparent_mov", "tooltip": "Output filename prefix without extension."},
                ),
                "save_path": (
                    "STRING",
                    {"default": "MatAnyone2", "tooltip": "Relative folder in ComfyUI/output, or an absolute directory."},
                ),
                "generate_video_preview": ("BOOLEAN", {"default": False}),
                "save_preview_image": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "alpha_mask": ("MASK",),
                "audio_source": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_video"
    OUTPUT_NODE = True
    CATEGORY = MATANYONE2_CATEGORY

    def save_video(
        self,
        images,
        fps,
        filename_prefix,
        save_path,
        generate_video_preview,
        save_preview_image,
        alpha_mask=None,
        audio_source=None,
    ):
        import imageio_ffmpeg

        if images.ndim != 4 or images.shape[-1] < 3:
            raise RuntimeError(
                f"Expected IMAGE input with shape (B, H, W, C), got {tuple(images.shape)}"
            )

        image_width = int(images.shape[2])
        image_height = int(images.shape[1])
        full_output_folder, filename, counter, subfolder, external_output = resolve_output_location(
            filename_prefix, save_path, self.output_dir, image_width, image_height
        )

        alpha_source = alpha_mask if alpha_mask is not None else None

        output_file = f"{filename}_{counter:05}_.mov"
        output_path = os.path.join(full_output_folder, output_file)
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

        audio_input_args, audio_output_args = self._build_audio_args(audio_source, "pcm_s16le")
        cmd = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-nostats",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-s",
            f"{image_width}x{image_height}",
            "-r",
            str(float(fps)),
            "-i",
            "-",
            *audio_input_args,
            "-map",
            "0:v:0",
            "-c:v",
            "prores_ks",
            "-profile:v",
            "4",
            "-pix_fmt",
            "yuva444p10le",
            "-alpha_bits",
            "16",
            *audio_output_args,
            output_path,
        ]
        self._run_ffmpeg_stream(
            cmd,
            (
                np.concatenate(
                    [
                        self._get_frame_rgb_uint8(images, frame_idx),
                        (
                            self._get_frame_alpha_float(images, alpha_source, frame_idx)[..., None] * 255.0
                        )
                        .clip(0, 255)
                        .astype(np.uint8),
                    ],
                    axis=-1,
                )
                for frame_idx in range(images.shape[0])
            ),
            "Failed to save transparent MOV:",
        )

        results = []
        video_previews = []
        if generate_video_preview:
            preview_proxy_path = write_ui_proxy_preview_video(
                self.output_dir,
                f"{filename}_mov_preview",
                images,
                alpha_source,
                fps,
                audio_source=audio_source,
            )
            video_previews = ensure_ui_video_preview(preview_proxy_path, self.output_dir)
        if save_preview_image:
            preview_rgb = self._compose_frame(
                self._get_frame_rgb_uint8(images, 0),
                self._get_frame_alpha_float(images, alpha_source, 0),
                np.array([0.0, 0.0, 0.0], dtype=np.float32),
            )
            preview_file = write_preview_png(full_output_folder, filename, counter, preview_rgb)
            if external_output:
                ui_prefix = build_filename_prefix(f"{filename}_preview_cache", "MatAnyone2/previews")
                ui_folder, ui_filename, ui_counter, ui_subfolder, _ = folder_paths.get_save_image_path(
                    ui_prefix, self.output_dir, image_width, image_height
                )
                ui_preview_file = write_preview_png(ui_folder, ui_filename, ui_counter, preview_rgb)
                results.append({"filename": ui_preview_file, "subfolder": ui_subfolder, "type": self.type})
            else:
                results.append({"filename": preview_file, "subfolder": subfolder, "type": self.type})

        ui = {"gifs": video_previews}
        if results:
            ui["images"] = results
        return {"ui": ui}


NODE_CLASS_MAPPINGS = {
    "LoadVideoForMatAnyone2": LoadVideoForMatAnyone2,
    "MatAnyone2": MatAnyone2Node,
    "SaveMatAnyone2Video": SaveMatAnyone2Video,
    "SaveMatAnyone2TransparentWebM": SaveMatAnyone2TransparentWebM,
    "SaveMatAnyone2TransparentMOV": SaveMatAnyone2TransparentMOV,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadVideoForMatAnyone2": "Load Video For MatAnyone2",
    "MatAnyone2": "MatAnyone2 Video Matting",
    "SaveMatAnyone2Video": "Save MatAnyone2 Video",
    "SaveMatAnyone2TransparentWebM": "Save MatAnyone2 Transparent WebM",
    "SaveMatAnyone2TransparentMOV": "Save MatAnyone2 Transparent MOV",
}
