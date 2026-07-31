"""Per-run image fixtures with exact, judge-free acceptance criteria."""

from __future__ import annotations

import base64
import hashlib
import io
import re
import secrets
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
_COLORS: dict[str, tuple[int, int, int]] = {
    "red": (220, 0, 0),
    "blue": (0, 80, 220),
    "green": (0, 170, 0),
    "orange": (255, 128, 0),
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
            "Read the qualification card. Reply exactly in this format: "
            "COLOR SHAPE | CODE. Use the pictured color, pictured shape, and "
            "complete large six-character code. Choose COLOR from red, blue, "
            "green, or orange, and SHAPE from circle, diamond, or triangle."
        )

    def response_matches(self, response: str) -> tuple[bool, bool]:
        """Check the exact code and visual attribute without a judge model."""

        code_matched, color_matched, shape_matched = self.response_match_details(
            response
        )
        return code_matched, color_matched and shape_matched

    def response_match_details(self, response: str) -> tuple[bool, bool, bool]:
        """Check the code, color, and shape independently."""

        normalized = " ".join(response.upper().replace("-", " ").split())
        code_matched = _code_matches(self.code, response)
        color_matched = self.color.upper() in normalized
        shape_matched = self.shape.upper() in normalized
        return code_matched, color_matched, shape_matched

    def response_format_matches(self, response: str) -> bool:
        """Require an exact final answer within a bounded response.

        Merely mentioning the hidden attributes somewhere in thousands of
        looping tokens is not a successful user response. A small model may
        include a bounded explanatory preamble despite the prompt, so accept
        that when the last non-empty line is the exact answer. Reject an
        immediately duplicated final answer while allowing the model to reason
        about the same attributes earlier in its bounded response. Markdown
        emphasis, harmless terminal punctuation, and visual grouping inside
        the code remain accepted.
        """

        if len(response) > 1600:
            return False
        separator = r"[\s-]*"
        grouped_code = separator.join(re.escape(character) for character in self.code)
        pattern = (
            rf"\s*[`*_\"']*\s*{re.escape(self.color)}\s+"
            rf"{re.escape(self.shape)}\s*\|\s*{grouped_code}"
            r"\s*[`*_\"'.!]*\s*"
        )
        lines = [line for line in response.splitlines() if line.strip()]
        if not lines or re.fullmatch(pattern, lines[-1], flags=re.IGNORECASE) is None:
            return False
        return not (
            len(lines) > 1
            and re.fullmatch(pattern, lines[-2], flags=re.IGNORECASE) is not None
        )

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


def _code_matches(code: str, response: str) -> bool:
    """Match every exact code character while tolerating visual grouping.

    Small vision models sometimes insert spaces or hyphens between groups of
    correctly read characters. Those separators are presentation rather than
    OCR content. Boundaries and the full ordered character sequence remain
    mandatory, so substitutions, omissions, additions, and repetitions still
    fail qualification.
    """

    separator = r"[\s-]*"
    exact_grouped_code = separator.join(re.escape(character) for character in code)
    pattern = rf"(?<![A-Z0-9]){exact_grouped_code}(?![A-Z0-9])"
    return re.search(pattern, response.upper()) is not None


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
        draw.ellipse((370, 270, 530, 430), fill=color, outline=outline, width=7)
    elif shape == "diamond":
        draw.polygon(
            ((450, 270), (530, 350), (450, 430), (370, 350)),
            fill=color,
            outline=outline,
        )
        draw.line(
            ((450, 270), (530, 350), (450, 430), (370, 350), (450, 270)),
            fill=outline,
            width=7,
        )
    else:
        draw.polygon(
            ((450, 270), (542, 430), (358, 430)),
            fill=color,
            outline=outline,
        )
        draw.line(
            ((450, 270), (542, 430), (358, 430), (450, 270)),
            fill=outline,
            width=7,
        )
