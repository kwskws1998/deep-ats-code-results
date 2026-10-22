"""Bounded encoder batches over the union of explicitly acquired video frames."""

import numpy as np
import torch


class EncodedFrames:
    """Share query-independent features within one update or static evaluation.

    Only union-mask frames enter the encoder. A caller must still gate each
    query's access with its own observation mask. No persistent training graph
    or features survive an optimizer update.
    """

    def __init__(self, model, frames, acquired, chunk_size=8):
        if chunk_size < 1 or acquired.shape != (len(frames),) or acquired.dtype != bool:
            raise ValueError("Need a positive encoder chunk and a boolean frame union.")
        self.features = {}
        self.encoded_frames = int(acquired.sum())
        self.encoder_calls = 0
        self.transfer_batches = 0
        clocks = np.flatnonzero(acquired)
        for offset in range(0, len(clocks), chunk_size):
            selected = clocks[offset : offset + chunk_size]
            images = (
                torch.as_tensor(
                    np.ascontiguousarray(frames[selected]),
                    device=model.transition.device,
                )
                .permute(0, 3, 1, 2)
                .contiguous()
                .float()
                / 255
            )
            values = model.frame_features(images)
            self.features.update(zip(selected.tolist(), values.split(1)))
            self.encoder_calls += 1
            self.transfer_batches += 1

    def __getitem__(self, clock):
        """Reject any request outside the declared acquisition union."""
        return self.features[int(clock)]
