Fixed
^^^^^

* Enabled self-collision at both the URDF importer and articulation levels, fixed the canonical
  right-leg/left-leg action and observation order, and replaced the former uniform joint action
  scale with per-joint position scales. Existing checkpoints must be retrained; overlapping
  STL convex hulls still require dedicated collision-mesh validation.

Added
^^^^^

* Added optional sensor-reference playback and joint/velocity/IMU/contact imitation reward
  support for calibrated ``.npz``, ``.pt/.pth``, ``.json`` and ``.csv`` recordings.
