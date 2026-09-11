"""Owner: Student 3

Image upload, preview and basic input-quality checks. Knows nothing about the
model: it only produces the RGB array the backend expects, plus a list of
plain-language issues to show the farmer.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import streamlit as st
from PIL import Image, ImageStat, UnidentifiedImageError

ACCEPTED_TYPES = ["jpg", "jpeg", "png"]

# Quality thresholds. Tune these once the real dataset statistics are known.
MIN_SIDE_BLOCKING = 64  # below this the image is unusable
MIN_SIDE_WARNING = 224  # below this the model input is heavily upscaled
MAX_ASPECT_RATIO = 3.2  # very wide/tall crops usually mean poor framing
DARK_MEAN_THRESHOLD = 45  # average brightness, 0-255
BRIGHT_MEAN_THRESHOLD = 228
LOW_CONTRAST_STD = 18  # flat image: fish not separable from the water
MAX_FILE_MB = 10

INVALID_IMAGE_MESSAGE = "Please upload a valid JPG/PNG image."


@dataclass
class UploadResult:
    """Everything app.py needs to know about the current upload."""

    array: np.ndarray | None = None
    image: Image.Image | None = None
    file_id: str | None = None  # changes whenever a different file is chosen
    filename: str | None = None
    issues: list[tuple[str, str]] = field(default_factory=list)  # (level, text)

    @property
    def has_image(self) -> bool:
        return self.array is not None

    @property
    def is_blocked(self) -> bool:
        """True when the image cannot be analysed at all."""
        return any(level == "error" for level, _ in self.issues)


def _check_quality(image: Image.Image) -> list[tuple[str, str]]:
    """Cheap PIL-only checks. Sharpness/blur detection can be added here."""
    issues: list[tuple[str, str]] = []
    width, height = image.size

    # Extremely small images.
    if min(width, height) < MIN_SIDE_BLOCKING:
        issues.append(
            (
                "error",
                f"This image is too small ({width}x{height} px) to analyse. Please "
                "upload a photo of at least 224x224 pixels.",
            )
        )
        return issues  # no point running the rest

    if min(width, height) < MIN_SIDE_WARNING:
        issues.append(
            (
                "warning",
                f"Low resolution ({width}x{height} px). The result may be less "
                "reliable - a closer, larger photo works better.",
            )
        )

    # Poor framing.
    ratio = max(width, height) / max(1, min(width, height))
    if ratio > MAX_ASPECT_RATIO:
        issues.append(
            (
                "warning",
                "The photo looks heavily cropped. Frame the whole fish in the "
                "centre of the picture for a better result.",
            )
        )

    try:
        stats = ImageStat.Stat(image.convert("L"))
        mean = stats.mean[0]
        stddev = stats.stddev[0]
    except Exception:  # pragma: no cover - never block an upload on statistics
        return issues

    exposure_flagged = True
    if mean < DARK_MEAN_THRESHOLD:
        issues.append(
            (
                "warning",
                "This photo is very dark. Retake it in daylight or with better "
                "lighting so the skin and fins are clearly visible.",
            )
        )
    elif mean > BRIGHT_MEAN_THRESHOLD:
        issues.append(
            (
                "warning",
                "This photo is over-exposed. Avoid direct flash or glare on wet "
                "skin, then take it again.",
            )
        )
    else:
        exposure_flagged = False

    # A dark or blown-out photo is low-contrast by definition; only raise this
    # separately when the exposure itself was fine.
    if not exposure_flagged and stddev < LOW_CONTRAST_STD:
        issues.append(
            (
                "warning",
                "The fish is hard to separate from the background. Place it on a "
                "plain surface and take the photo again.",
            )
        )

    return issues


def _open_image(raw: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(raw))
    image.load()  # decode now, so corrupt files fail here rather than later
    return image.convert("RGB")


def render_upload_card() -> UploadResult:
    """Draw the upload card and return the validated image plus its issues."""
    st.markdown(
        '<div class="ah-section-title">Step 1 &middot; Upload a fish photo</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload a clear photo of one fish (JPG or PNG)",
        type=ACCEPTED_TYPES,
        accept_multiple_files=False,
        help="Best results: daylight, plain background, whole fish in frame.",
        key="ah_uploader",
    )

    if uploaded is None:
        st.markdown(
            '<div class="ah-msg info">&#128247; <strong>Tip:</strong> hold the fish '
            "flat, take the photo from the side in daylight, and make sure the head, "
            "body and fins are all in the picture.</div>",
            unsafe_allow_html=True,
        )
        return UploadResult()

    raw = uploaded.getvalue()
    # An id that changes whenever a different file is picked, so app.py can
    # drop any stale prediction.
    file_id = f"{uploaded.name}:{len(raw)}"

    if not raw:
        return UploadResult(
            file_id=file_id,
            filename=uploaded.name,
            issues=[("error", INVALID_IMAGE_MESSAGE)],
        )

    size_mb = len(raw) / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        return UploadResult(
            file_id=file_id,
            filename=uploaded.name,
            issues=[
                (
                    "error",
                    f"This file is {size_mb:.1f} MB. Please upload an image under "
                    f"{MAX_FILE_MB} MB.",
                )
            ],
        )

    try:
        image = _open_image(raw)
    except (UnidentifiedImageError, OSError, ValueError):
        return UploadResult(
            file_id=file_id,
            filename=uploaded.name,
            issues=[("error", INVALID_IMAGE_MESSAGE)],
        )

    result = UploadResult(
        array=np.array(image),
        image=image,
        file_id=file_id,
        filename=uploaded.name,
        issues=_check_quality(image),
    )
    _render_preview(result)
    return result


def render_upload() -> np.ndarray | None:
    """Backward-compatible entry point: the RGB array, or None.

    Kept so any caller written against the original skeleton keeps working.
    """
    return render_upload_card().array


def _render_preview(result: UploadResult) -> None:
    """Preview the image alongside any quality feedback."""
    if result.image is None:
        return

    width, height = result.image.size
    preview_col, info_col = st.columns([1, 1], gap="large")

    with preview_col:
        st.image(result.image, use_container_width=True)
        st.markdown(
            f'<div class="ah-preview-meta">{result.filename} &middot; {width}x{height} px</div>',
            unsafe_allow_html=True,
        )

    with info_col:
        if result.is_blocked:
            st.markdown('<div class="ah-eyebrow">Cannot analyse</div>', unsafe_allow_html=True)
        elif result.issues:
            st.markdown('<div class="ah-eyebrow">Image quality notes</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="ah-eyebrow">Image check</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="ah-msg ok">&#9989; This photo looks good. You can run '
                "the analysis.</div>",
                unsafe_allow_html=True,
            )

        for level, text in result.issues:
            css = "err" if level == "error" else "warn"
            icon = "&#9940;" if level == "error" else "&#9888;&#65039;"
            st.markdown(f'<div class="ah-msg {css}">{icon} {text}</div>', unsafe_allow_html=True)
