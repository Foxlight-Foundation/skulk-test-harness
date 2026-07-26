"""The per-run phrase a qualification asks a model to repeat back.

Shared by the browser journey and the direct API check so the two cannot
drift. The phrase deliberately reads as ordinary language: it used to be a
hex nonce of the shape ``FRESH-3D8C32F7`` / ``API-3D8C32F7``, which reads as
a credential, and a safety-tuned 1B model refused to repeat one at all:

    I can't assist with generating or distributing cryptographic resources
    that could be used for malicious purposes.

That failed a qualification leg whose install, download, launch, streaming,
and rendering had every one of them worked. Later verbatim-copy instructions
triggered the same kind of refusal even with ordinary words. The check only
needs unpredictable terms to survive the whole request and response path, so a
benign creative-writing task proves that without asking the model to reproduce
credential-like or extracted text.

The task also has to be concrete. Asking for merely "one friendly sentence"
left a 1B instruct model unsure what kind of sentence was wanted, so it asked a
clarifying question despite receiving and rendering the request correctly. A
short hotel-welcome scenario gives even a small model enough intent to answer
while preserving the same randomized response proof.
"""

from __future__ import annotations

import re
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

def echo_phrase() -> str:
    """Return a benign, unpredictable phrase for an echo assertion.

    Two distinct common words plus a four digit number. Unpredictable enough
    that a stale or replayed response cannot satisfy the check, and plainly
    not a credential.
    """

    first, second = secrets.SystemRandom().sample(_ECHO_WORDS, 2)
    return f"{first} {second} {secrets.randbelow(9000) + 1000}"


def echo_prompt(phrase: str) -> str:
    """Return a concrete, natural writing task containing every random item."""

    first, second, room = phrase.split()
    return (
        "Write one short, friendly welcome sentence for a hotel guest. "
        f"The hotel is named {first.title()} {second.title()}, and the guest's "
        f"room number is {room}. Mention both {first.title()} {second.title()} "
        f"and room {room} in your sentence."
    )


def echo_matched(phrase: str, response: str) -> bool:
    """Report whether ``response`` contains every unpredictable phrase item.

    The creative-writing prompt need not keep the terms adjacent. Requiring
    each complete, case-insensitive word still prevents a stale response from
    satisfying the check while allowing natural punctuation and prose.
    """

    return all(
        re.search(rf"(?<!\w){re.escape(item)}(?!\w)", response, re.IGNORECASE)
        is not None
        for item in phrase.split()
    )
