"""Packed-token layout metadata for MiniMax H3.

Core builds `comfy.ldm.minimax.model.PackedLayout` per sampling run and keeps it
in the minimax payload, but only as a flat `(start, stop, kind)` segment table.
This module turns that table into the named ranges the attention probe needs:

    {
        "text_range": (start, end),
        "reference_ranges": [...],
        "audio_range": (start, end),
        "video_range": (start, end),
        "video_shape": (latent_t, patch_h, patch_w),
    }

Nothing here mutates model state; it only reads a layout that already exists.
"""

from dataclasses import dataclass, field

# segment kinds emitted by PackedLayout, in packing order
KIND_TEXT = "text"
KIND_COND = "cond"
KIND_REF_IMG = "ref_img"
KIND_REF_AUDIO = "ref_audio"
KIND_AUDIO = "audio"
KIND_VIDEO = "video"

CONTEXT_KINDS = (KIND_COND, KIND_REF_IMG, KIND_REF_AUDIO)


@dataclass
class TokenLayout:
    """Named token ranges over one packed H3 sequence.

    All ranges are half-open `(start, stop)` row indices into the packed
    sequence that attention sees, i.e. directly usable to slice Q/K.
    """

    seq_len: int
    text_range: tuple
    audio_range: tuple
    video_range: tuple
    video_shape: tuple                      # (latent_t, patch_h, patch_w)
    audio_t: int                            # latent audio frames; rows are 2 * audio_t (channel-major)
    reference_ranges: list = field(default_factory=list)   # [(kind, start, stop)]
    segments: list = field(default_factory=list)           # [(start, stop, kind)] verbatim

    @property
    def frame_rows(self):
        """Video rows per latent frame (patch_h * patch_w)."""
        return self.video_shape[1] * self.video_shape[2]

    @property
    def latent_t(self):
        return self.video_shape[0]

    def video_frame_range(self, t):
        """Row range of one target-video latent frame."""
        v0 = self.video_range[0]
        n = self.frame_rows
        return v0 + t * n, v0 + (t + 1) * n

    def frame_of_row(self, row):
        """Latent frame index for a target-video row, or None if outside video."""
        v0, v1 = self.video_range
        if not (v0 <= row < v1):
            return None
        return (row - v0) // self.frame_rows

    def context_ranges(self):
        """Every non-target range: text plus keyframe/reference conditioning."""
        return [(KIND_TEXT,) + self.text_range] + list(self.reference_ranges)

    def as_dict(self):
        return {
            "text_range": self.text_range,
            "reference_ranges": list(self.reference_ranges),
            "audio_range": self.audio_range,
            "video_range": self.video_range,
            "video_shape": self.video_shape,
            "seq_len": self.seq_len,
            "audio_t": self.audio_t,
        }

    def describe(self):
        parts = [
            "seq_len=%d" % self.seq_len,
            "text=%d" % (self.text_range[1] - self.text_range[0]),
        ]
        for kind, a, b in self.reference_ranges:
            parts.append("%s=%d" % (kind, b - a))
        parts.append("audio=%d(t=%d)" % (self.audio_range[1] - self.audio_range[0], self.audio_t))
        t, ph, pw = self.video_shape
        parts.append("video=%d(t=%d,%dx%d)" % (self.video_range[1] - self.video_range[0], t, ph, pw))
        return " ".join(parts)


def from_packed_layout(packed):
    """Build a TokenLayout from a core `PackedLayout`.

    The signature is `(text_len, latent_t, latent_h, latent_w, audio_t)` in
    *latent* units; the DiT's 1x2x2 patching halves h and w, so a target video
    frame occupies `(latent_h // 2) * (latent_w // 2)` rows.
    """
    text_len, latent_t, latent_h, latent_w, audio_t = packed.signature
    segments = [(a, b, k) for a, b, k in packed.segments]

    text_range = None
    audio_range = None
    video_range = None
    refs = []
    for a, b, kind in segments:
        if kind == KIND_TEXT:
            text_range = (a, b)
        elif kind == KIND_VIDEO:
            video_range = (a, b)          # target video: always the final segment
        elif kind == KIND_AUDIO:
            audio_range = (a, b)          # target audio: always immediately before it
        elif kind in CONTEXT_KINDS:
            refs.append((kind, a, b))

    if text_range is None or audio_range is None or video_range is None:
        raise ValueError("unexpected H3 packed layout: missing text/audio/video segment")

    return TokenLayout(
        seq_len=packed.seq_len,
        text_range=text_range,
        audio_range=audio_range,
        video_range=video_range,
        video_shape=(latent_t, latent_h // 2, latent_w // 2),
        audio_t=audio_t,
        reference_ranges=refs,
        segments=segments,
    )


def resolve_layout(x, context, payload):
    """Get the layout for a forward pass, rebuilding it only if core did not.

    `MiniMaxH3.extra_conds` normally prebuilds `payload["layout"]` once per
    sampling run. Core revalidates it against the actual shapes and rebuilds on
    mismatch; this mirrors that check so the probe never labels tokens with a
    stale layout.
    """
    import comfy.ldm.common_dit
    from comfy.ldm.minimax.model import PackedLayout

    video_x, audio_x = x[0], x[1]
    patch = (1, 2, 2)
    video_x = comfy.ldm.common_dit.pad_to_patch_size(video_x, patch)
    latent_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
    audio_t = audio_x.shape[-1]
    text_len = context.shape[1]
    signature = (text_len, latent_t, lat_h, lat_w, audio_t)

    packed = (payload or {}).get("layout")
    if packed is None or packed.signature != signature:
        packed = PackedLayout(
            text_len, latent_t, lat_h, lat_w, audio_t,
            keyframes=(payload or {}).get("keyframes"),
            refs=(payload or {}).get("refs"),
        )
    return from_packed_layout(packed)
