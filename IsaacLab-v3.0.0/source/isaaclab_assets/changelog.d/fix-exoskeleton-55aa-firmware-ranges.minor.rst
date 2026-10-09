Changed
^^^^^^^

* **Breaking:** Matched the eight-motor ``55 AA`` wire format to the firmware's
  big-endian fields, fixed position range of ``[-12.57, 12.57] rad``, signed
  velocity range of ``[-20, 20] rad/s``, and unsigned acceleration range of
  ``[0, 65.535] rad/s^2``. Configuration schema changed to
  ``exoskeleton-55aa-config/v2``; migrate calibration files using the new
  template and regenerate binary output. Existing non-empty binary files
  were rejected as output paths to prevent mixed incompatible recordings.
  The eight-joint action/checkpoint and JSONL contracts remained unchanged.
