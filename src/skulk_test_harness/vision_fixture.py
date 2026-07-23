"""Per-run image fixtures with exact, judge-free acceptance criteria."""

from __future__ import annotations

import base64
import hashlib
import io
import secrets
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
_COLORS: dict[str, tuple[int, int, int]] = {
    "amber": (245, 158, 11),
    "cyan": (6, 182, 212),
    "magenta": (217, 70, 239),
    "lime": (132, 204, 22),
}
_SHAPES = ("circle", "diamond", "triangle")


@dataclass(frozen=True)
class VisionFixture:
    """Generated PNG plus hidden exact-answer attributes."""

    png: bytes
    code: str
    color: str
    shape: str

    @property
    def sha256(self) -> str:
        """Return the fixture byte digest."""

        return hashlib.sha256(self.png).hexdigest()

    @property
    def code_sha256(self) -> str:
        """Return a safe digest of the hidden code for reports."""

        return hashlib.sha256(self.code.encode()).hexdigest()

    @property
    def data_url(self) -> str:
        """Return an OpenAI-compatible PNG data URL."""

        encoded = base64.b64encode(self.png).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    @property
    def prompt(self) -> str:
        """Return a prompt that never embeds the hidden answer."""

        return (
            "Inspect the attached qualification card. Reply with the large "
            "six-character code, then the colored shape as '<color> <shape>'."
        )

    def response_matches(self, response: str) -> tuple[bool, bool]:
        """Check the exact code and visual attribute without a judge model."""

        normalized = " ".join(response.upper().replace("-", " ").split())
        code_matched = self.code in normalized
        attribute_matched = (
            self.color.upper() in normalized and self.shape.upper() in normalized
        )
        return code_matched, attribute_matched

    def write(self, path: Path) -> None:
        """Persist the private fixture for later inspection."""

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.png)
        path.chmod(0o600)


def generate_vision_fixture() -> VisionFixture:
    """Generate a high-contrast randomized qualification card."""

    code = "".join(secrets.choice(_ALPHABET) for _ in range(6))
    color = secrets.choice(tuple(_COLORS))
    shape = secrets.choice(_SHAPES)
    image = Image.new("RGB", (900, 540), color=(248, 250, 252))
    draw = ImageDraw.Draw(image)
    font = _load_font(154)
    small_font = _load_font(42)
    draw.rounded_rectangle(
        (28, 28, 872, 512),
        radius=28,
        fill=(255, 255, 255),
        outline=(15, 23, 42),
        width=8,
    )
    code_box = draw.textbbox((0, 0), code, font=font)
    code_width = code_box[2] - code_box[0]
    draw.text(
        ((900 - code_width) / 2, 76),
        code,
        font=font,
        fill=(15, 23, 42),
    )
    _draw_shape(draw, shape, _COLORS[color])
    draw.text(
        (56, 452),
        "FRESH INSTALL VISION QUALIFICATION",
        font=small_font,
        fill=(71, 85, 105),
    )
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=False)
    return VisionFixture(
        png=output.getvalue(),
        code=code,
        color=color,
        shape=shape,
    )


def data_url_sha256(data_url: str) -> str:
    """Decode a base64 image data URL and return its byte digest."""

    prefix, separator, encoded = data_url.partition(",")
    if not separator or ";base64" not in prefix or not prefix.startswith("data:image/"):
        raise ValueError("expected a base64 image data URL")
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except ValueError as exception:
        raise ValueError("invalid base64 image data URL") from exception
    return hashlib.sha256(decoded).hexdigest()


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Use a common bold font when present and Pillow's bundled fallback otherwise."""

    for path in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _draw_shape(
    draw: ImageDraw.ImageDraw,
    shape: str,
    color: tuple[int, int, int],
) -> None:
    """Draw the randomized visual attribute in a fixed high-contrast region."""

    outline = (15, 23, 42)
    if shape == "circle":
        draw.ellipse((350, 280, 550, 440), fill=color, outline=outline, width=7)
    elif shape == "diamond":
        draw.polygon(
            ((450, 270), (570, 360), (450, 450), (330, 360)),
            fill=color,
            outline=outline,
        )
        draw.line(
            ((450, 270), (570, 360), (450, 450), (330, 360), (450, 270)),
            fill=outline,
            width=7,
        )
    else:
        draw.polygon(
            ((450, 270), (570, 440), (330, 440)),
            fill=color,
            outline=outline,
        )
        draw.line(
            ((450, 270), (570, 440), (330, 440), (450, 270)),
            fill=outline,
            width=7,
        )
