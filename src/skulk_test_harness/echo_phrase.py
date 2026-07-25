"""The per-run phrase a qualification asks a model to repeat back.

Shared by the browser journey and the direct API check so the two cannot
drift. The phrase deliberately reads as ordinary language: it used to be a
hex nonce of the shape ``FRESH-3D8C32F7`` / ``API-3D8C32F7``, which reads as
a credential, and a safety-tuned 1B model refused to repeat one at all:

    I can't assist with generating or distributing cryptographic resources
    that could be used for malicious purposes.

That failed a qualification leg whose install, download, launch, streaming,
and rendering had every one of them worked. The check only needs a phrase the
run picks unpredictably and the response repeats, so looking like a secret
buys nothing and costs a false failure on exactly the small aligned models a
fresh install is most likely to start with.
"""

from __future__ import annotations

import secrets

_ECHO_WORDS = (
    "amber",
    "harbor",
    "willow",
    "cobalt",
    "meadow",
    "lantern",
    "copper",
    "juniper",
    "quartz",
    "thicket",
    "saffron",
    "beacon",
)

ECHO_INSTRUCTION = "Repeat this phrase back exactly and say nothing else: "


def echo_phrase() -> str:
    """Return a benign, unpredictable phrase for an echo assertion.

    Two distinct common words plus a four digit number. Unpredictable enough
    that a stale or replayed response cannot satisfy the check, and plainly
    not a credential.
    """

    first, second = secrets.SystemRandom().sample(_ECHO_WORDS, 2)
    return f"{first} {second} {secrets.randbelow(9000) + 1000}"


def echo_prompt(phrase: str) -> str:
    """Return the full user message asking a model to repeat ``phrase``."""

    return ECHO_INSTRUCTION + phrase


def echo_matched(phrase: str, response: str) -> bool:
    """Report whether ``response`` repeats ``phrase``.

    Case-insensitive: a model that capitalizes the start of its reply has
    still demonstrated the whole chat path works, which is what this check
    exists to prove.
    """

    return phrase.upper() in response.upper()
