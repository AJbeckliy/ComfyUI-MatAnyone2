import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

function fitHeight(node) {
    node.setSize([node.size[0], node.computeSize([node.size[0], node.size[1]])[1]]);
    node?.graph?.setDirtyCanvas(true, true);
}

function installPreview(nodeType) {
    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        originalOnNodeCreated?.apply(this, arguments);

        const root = document.createElement("div");
        const previewWidget = this.addDOMWidget("videopreview", "preview", root, {
            serialize: false,
            hideOnZoom: false,
            getValue() {
                return root.value;
            },
            setValue(v) {
                root.value = v;
            },
        });

        previewWidget.parentEl = document.createElement("div");
        previewWidget.parentEl.className = "matanyone2_video_preview";
        previewWidget.parentEl.style.width = "100%";
        previewWidget.parentEl.style.marginTop = "8px";
        previewWidget.parentEl.style.display = "none";
        root.appendChild(previewWidget.parentEl);

        previewWidget.computeSize = function (width) {
            if (this.parentEl.style.display !== "none" && this.totalHeight > 0) {
                return [width, this.totalHeight];
            }
            return [width, -4];
        };

        previewWidget.totalHeight = 0;
    };

    const originalOnExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
        originalOnExecuted?.apply(this, arguments);

        const previewWidget = this.widgets?.find((w) => w.name === "videopreview");
        if (!previewWidget) return;

        previewWidget.parentEl.replaceChildren();
        previewWidget.totalHeight = 0;

        const gifs = message?.gifs || [];
        if (!gifs.length) {
            previewWidget.parentEl.style.display = "none";
            fitHeight(this);
            return;
        }

        const item = gifs.find((gif) => gif && gif.filename);
        if (!item) {
            previewWidget.parentEl.style.display = "none";
            fitHeight(this);
            return;
        }

        const video = document.createElement("video");
        video.controls = true;
        video.loop = true;
        video.defaultMuted = true;
        video.muted = true;
        video.autoplay = true;
        video.playsInline = true;
        video.style.width = "100%";
        video.style.borderRadius = "8px";
        video.style.background = "#111";

        const params = new URLSearchParams({
            filename: item.filename,
            subfolder: item.subfolder || "",
            type: item.type || "output",
            t: Date.now().toString(),
        });
        video.src = api.apiURL("/view?" + params.toString());

        video.addEventListener("loadedmetadata", () => {
            previewWidget.parentEl.style.display = "block";
            previewWidget.totalHeight = Math.max(220, Math.round((this.size[0] - 20) * 0.6));
            fitHeight(this);
        });

        video.addEventListener("error", () => {
            previewWidget.parentEl.style.display = "none";
            fitHeight(this);
        });

        previewWidget.parentEl.appendChild(video);
        previewWidget.parentEl.style.display = "block";
        fitHeight(this);
    };
}

app.registerExtension({
    name: "ComfyUI.MatAnyone2.VideoPreview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (
            nodeData.name === "SaveMatAnyone2Video" ||
            nodeData.name === "SaveMatAnyone2TransparentWebM" ||
            nodeData.name === "SaveMatAnyone2TransparentMOV"
        ) {
            installPreview(nodeType);
        }
    },
});
