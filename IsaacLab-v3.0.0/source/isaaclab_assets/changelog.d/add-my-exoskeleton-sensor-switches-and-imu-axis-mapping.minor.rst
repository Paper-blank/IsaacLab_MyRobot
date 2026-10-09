Added
^^^^^

* Added train/play switches for selectively enabling IMU, RGB, event, and contact sensors. Disabled
  channels are omitted from the scene but retain fixed observation shapes and invalid masks so PPO
  network dimensions remain stable.

* Added explicit dataset-to-joint IMU axis calibration for the mirrored left and right leg sensors.
  IMU-01/03/05 use ``[-y, x, -z]`` and IMU-02/04/06 use ``[y, x, -z]``.  The recorder positions are
  01/02 ankles, 03/04 knees, and 05/06 hips; existing checkpoints should
  be retrained when this calibration changes the reference target convention.
