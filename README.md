# ComfyUI-MatAnyone2

`ComfyUI-MatAnyone2` 是一个面向 `MatAnyone2` 视频抠像流程的 `ComfyUI` 自定义节点插件。

它不是只把模型简单封装成一个节点，而是尽量把实际使用里需要的整条链路打通：

- 加载视频
- 选择遮罩参考帧 `mask_frame`
- 预览并导出该帧用于制作 mask
- 执行 `MatAnyone2` 视频抠像
- 导出普通 `MP4`、透明 `WebM` 或透明 `MOV`
- 接入原视频音频并随输出自动保留

## 功能特点

- 独立的 `MatAnyone2` 节点分类
- `Load Video For MatAnyone2` 支持视频输入、帧数统计、FPS、音频路径输出、遮罩参考帧预览
- `MatAnyone2 Video Matting` 支持 `mask_frame_index`
- `Save MatAnyone2 Video` 用于导出普通成品 `MP4`
- `Save MatAnyone2 Transparent WebM` 用于导出透明 `VP9 WebM`
- `Save MatAnyone2 Transparent MOV` 用于导出透明 `ProRes 4444 MOV`
- 保存节点内置视频预览
- 模型统一存放在 `ComfyUI/models/matanyone2`，不依赖系统缓存目录

## 适用场景

这个插件更适合“视频目标抠像”场景，不是通用的一键单图自动抠图工具。

如果你的工作流是：

- 已经知道要抠谁
- 愿意在某一帧提供一个前景遮罩
- 希望模型把整段视频稳定传播出来

那这套节点会比较适合。

## 安装方法

将仓库克隆到 `ComfyUI/custom_nodes` 目录：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/AJbeckliy/ComfyUI-MatAnyone2.git
```

在 `ComfyUI` 使用的同一个 Python 环境里安装依赖：

```bash
cd ComfyUI-MatAnyone2
pip install -r requirements.txt
```

安装完成后重启 `ComfyUI`。

## 模型位置

插件会自动使用或下载模型到以下目录：

```text
ComfyUI/models/matanyone2/MatAnyone2
ComfyUI/models/matanyone2/torch_hub/checkpoints
```

手动下载模型：

- 夸克网盘: [https://pan.quark.cn/s/c3d88e3f9ece](https://pan.quark.cn/s/c3d88e3f9ece)

如果你手动下载模型，请将文件放到：

```text
ComfyUI/models/matanyone2/
```

如果压缩包里还包含下面这些 ResNet 权重：

- `resnet18-5c106cde.pth`
- `resnet50-19c8e357.pth`

请放到：

```text
ComfyUI/models/matanyone2/torch_hub/checkpoints
```

## 节点说明

### Load Video For MatAnyone2

输出：

- `frames`
- `mask_frame_image`
- `frame_count`
- `mask_frame_index`
- `fps`
- `audio_source`

其中 `mask_frame` 用于指定哪一帧作为遮罩参考帧，同时：

- `mask_frame_image` 会输出该帧图像
- `mask_frame_index` 会把这个索引传给后续抠像节点

### MatAnyone2 Video Matting

输入：

- `images`
- `mask_frame_index`
- `invert_mask`
- `mask`

输出：

- `rgba_image`
- `alpha_mask`

说明：

- `mask` 是初始化前景遮罩
- `invert_mask` 用来处理遮罩黑白方向相反的问题
- `mask_frame_index` 决定模型从哪一帧开始向前后传播

### Save MatAnyone2 Video

用于导出普通 `MP4`：

- 会把透明前景与背景色合成后再输出
- 支持保留原视频音频
- 可选额外导出一条 `alpha` 黑白视频

### Save MatAnyone2 Transparent WebM

用于导出透明 `WebM`：

- 编码格式为 `VP9`
- 适合网页预览、轻量透明素材、透明视频测试

### Save MatAnyone2 Transparent MOV

用于导出透明 `MOV`：

- 编码格式为 `ProRes 4444`
- 更适合剪映、PR、FCP、Resolve 等后期软件
- 节点里的视频预览可以用代理视频显示，真实导出的仍然是透明 `MOV`

## 示例工作流

仓库内提供了 3 份完整示例：

- `example_video_workflow_full_pipeline.json`
- `example_video_workflow_transparent_webm.json`
- `example_video_workflow_transparent_mov.json`

这些工作流都包含：

- 视频输入
- 遮罩参考帧预览/导出
- MatAnyone2 抠像
- 最终视频导出

## 使用提醒

- `MatAnyone2` 是视频 matting 模型，不是通用的一键抠图模型
- 你仍然需要为选中的 `mask_frame` 提供前景遮罩
- 如果你输入的是单独的黑白 mask 图，通常保持 `invert_mask = false`
- 如果你使用的是某些由 alpha 推出来的遮罩，结果方向不对时可以尝试 `invert_mask = true`

## English Summary

`ComfyUI-MatAnyone2` is a practical `ComfyUI` node pack for `MatAnyone2` video matting.

It provides an end-to-end workflow inside ComfyUI:

- load a video
- choose a mask anchor frame
- preview/export that frame for mask creation
- run MatAnyone2 video matting
- export MP4, transparent WebM, or transparent MOV
- preserve original audio when `audio_source` is connected

## Credits

- [MatAnyone2](https://github.com/pq-yang/MatAnyone2)
- Bilibili: [https://space.bilibili.com/481121504?spm_id_from=333.1007.0.0](https://space.bilibili.com/481121504?spm_id_from=333.1007.0.0)
