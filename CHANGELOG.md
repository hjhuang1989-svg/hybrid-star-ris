# Changelog

## Task 1
- Rewrote Section IV-C as `Protocol-dependent extension to TS and SS`.
- Removed the redundant ES recap because ES is already defined by Eqs. (13)-(14).
- Compressed TS / SS into one protocol-indexed template using `p in {TS, SS}` and unified `(a_s^(p), b_s^(p), c_s^(p), d_s^(p))`.
- Kept only the protocol-specific distinctions needed for TS (slot-rate inflation) and SS (aperture shrinkage).

## Task 2
- Replaced the Introduction transition sentence with a formal journal-style lead-in.
- Removed informal/draft-facing wording.

## Task 3
- Rewrote contribution 3 so that it emphasizes the method structure rather than implementation detail.
- Removed `The released code` and the overly local formula-level phrasing.

## Task 4
- Updated the design-rules figure so that both subplots compare ES / TS / SS.
- Updated the hardware-sensitivity figure so that both subplots compare ES / TS / SS.
- The original file numbered the hardware-sensitivity content as Fig. 3 and the power-breakdown content as Fig. 4. For consistency with the requested content mapping, the revised manuscript places the power-breakdown figure before the hardware-sensitivity figure, so the hardware-sensitivity content is now Fig. 4.
- Extended the simulator to support ES / TS / SS for active-ratio extraction, gain extraction, amplifier-noise sweeps, and update-frequency sweeps.
