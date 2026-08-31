# H3 forecast representation interoperability

H3-Optimizations may change the internal target-video token order while keeping the external MiniMax H3 `_forward` contract unchanged. A denoiser forecast/cache implementation that bypasses `_forward` and feeds a synthesized final-block hidden state directly into H3 `FinalLayer` must therefore either consume the representation adapter described here or allow H3-Optimizations to use raster order.

## Consumer declaration

The current contract version is `1`.

A forecast consumer declares support in `transformer_options` under `h3_optimizations_forecast_consumers`, keyed by the consumer's stable wrapper key. Spectrum's current H3 diffusion wrapper key is `spectrum_minimax_h3`.

```python
transformer_options.setdefault(
    "h3_optimizations_forecast_consumers",
    {},
)["spectrum_minimax_h3"] = {
    "api": 1,
    "accepts_representation_adapter": True,
}
```

A known `_forward`-bypassing consumer that is active without this declaration causes H3-Optimizations to leave the target video in native raster order. Sparse attention remains active; only the optional non-raster token-order optimization is disabled.

## Producer contract

When H3-Optimizations installs a non-raster target-video representation it publishes `transformer_options["h3_optimizations_forecast_representation"]`:

```python
{
    "api": 1,
    "scope": "final_block_target_hidden",
    "representation": "cube_major_target_video",
    "native_representation": "raster_target_video",
    "video_token_order": (1, 8, 8),  # or another supported geometry
    "to_native_target_hidden": callable,
}
```

`to_native_target_hidden` converts a compact final-block target hidden tensor from the active H3-Optimizations representation to the native representation expected by H3 `FinalLayer`.

The v1 callable contract is:

```python
native_hidden = contract["to_native_target_hidden"](
    predicted_hidden,
    layout=layout,
    video_shape=video_x.shape,
    patch_size=inner.patch_size,
)
```

`predicted_hidden` is the compact `[..., target_audio_rows + target_video_rows, hidden]` tensor. The adapter preserves target-audio rows and restores only target-video rows to native raster order.

A forecast implementation may keep its history and regression/extrapolation state in the transformed representation. The required conversion point is immediately before a forecasted hidden state is passed into native H3 `FinalLayer` or another consumer that assumes raster target-video rows.

## Why the declaration is explicit

Shape and topology checks cannot distinguish raster and cube-major tensors with the same row count and hidden width. The consumer therefore opts in explicitly instead of H3-Optimizations assuming that an arbitrary cache or forecast wrapper understands the representation.

Future contract revisions can add other representations without requiring consumers to hard-code H3-Optimizations' ordering algorithms.
