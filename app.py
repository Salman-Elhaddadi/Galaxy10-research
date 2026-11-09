"""Galaxy10 clean Model A inference. Run: python -m streamlit run app.py."""
from pathlib import Path
import hashlib
import sys
import warnings

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
RUN_ID = "model-a-seed0-20261108T204027801805Z-9db1a3778b9b478aada60ddc853e86dc"
CHECKPOINT = ROOT / "checkpoints" / RUN_ID / "best.pt"
STATS = ROOT / "data" / "processed" / "train_normalization.npz"
CLASS_NAMES = (
    "Disturbed", "Merging", "Round Smooth", "In-between Round Smooth",
    "Cigar Shaped Smooth", "Barred Spiral", "Unbarred Tight Spiral",
    "Unbarred Loose Spiral", "Edge-on without Bulge", "Edge-on with Bulge",
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess(image, mean, std):
    import cv2
    import numpy as np
    import torch
    from galaxy10.data.preprocessing import normalize_image

    pixels = cv2.resize(np.asarray(image.convert("RGB")), (128, 128), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(pixels.transpose(2, 0, 1).copy()).float().div_(255.0)
    return normalize_image(tensor.clamp(0, 1), mean, std).unsqueeze(0)


@st.cache_resource
def load_artifacts(checkpoint_path, stats_path, checkpoint_hash, stats_hash, device):
    from galaxy10.data.preprocessing import load_normalization_stats
    from galaxy10.evaluate import load_checkpoint

    model, checkpoint = load_checkpoint(checkpoint_path, device)
    config = checkpoint["config"]
    if config.get("model") != "a" or config.get("condition") != "clean" or config.get("run_id") != RUN_ID:
        raise ValueError("Checkpoint does not identify the requested clean Model A run.")
    recorded = config.get("provenance", {}).get("file_sha256", {})
    recorded = {key.replace("\\", "/"): value for key, value in recorded.items()}
    expected = recorded.get("data/processed/train_normalization.npz")
    if not expected or expected != stats_hash:
        raise ValueError("Training normalization fingerprint is missing or does not match the checkpoint. Restore the matching file from the clean run archive.")
    mean, std = load_normalization_stats(stats_path)
    return model, mean, std


def predict(image, model, mean, std, device):
    import torch
    with torch.inference_mode():
        logits = model(preprocess(image, mean, std).to(device))
        if logits.shape != (1, 10) or not torch.isfinite(logits).all():
            raise ValueError("Model returned invalid scores.")
        return torch.softmax(logits, dim=1)[0].cpu().numpy()


def main():
    st.set_page_config(page_title="Galaxy10 classifier", page_icon="🌌")
    st.title("Galaxy10 morphology classifier")
    st.write("Upload a PNG or JPEG galaxy image to classify it with the clean Model A checkpoint.")
    try:
        import torch
        from PIL import Image, UnidentifiedImageError
        import cv2  # noqa: F401
    except ImportError as exc:
        st.error(f"Missing app dependency: {exc}. Install with `python -m pip install -e \".[app]\"`.")
        st.stop()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.caption(f"Model A · {device.upper()} · 128 × 128 RGB")
    missing = [str(path) for path in (CHECKPOINT, STATS) if not path.is_file()]
    if missing:
        st.error("Missing inference artifacts. Restore these files from galaxy10_clean_model_and_results_complete.zip:\n\n" + "\n\n".join(f"- `{path}`" for path in missing))
        st.stop()
    try:
        model, mean, std = load_artifacts(str(CHECKPOINT), str(STATS), sha256(CHECKPOINT), sha256(STATS), device)
    except Exception as exc:
        st.error(f"Cannot load inference artifacts: {exc}")
        st.stop()

    uploaded = st.file_uploader("Galaxy image", type=["png", "jpg", "jpeg"])
    if uploaded is None:
        st.info("Choose an image to see its classification.")
        return
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(uploaded) as source:
                if source.format not in {"PNG", "JPEG"} or getattr(source, "n_frames", 1) != 1:
                    raise ValueError("Please upload a single-frame PNG or JPEG image.")
                image = source.convert("RGB")
        st.image(image, caption="Uploaded image", width=400)
        probabilities = predict(image, model, mean, std, device)
        best = int(probabilities.argmax())
        st.subheader("Prediction")
        st.metric("Predicted class", CLASS_NAMES[best], f"{probabilities[best]:.2%} confidence", delta_color="off")
        st.subheader("All class probabilities")
        st.dataframe({"Class": CLASS_NAMES, "Probability": [f"{p:.2%}" for p in probabilities]}, hide_index=True, width="stretch")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        st.error(f"Cannot process this image: {exc}")
    except Exception as exc:
        st.error(f"Inference failed: {exc}")


if __name__ == "__main__":
    main()
