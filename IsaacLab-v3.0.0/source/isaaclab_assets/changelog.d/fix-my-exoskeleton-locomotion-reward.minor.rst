Fixed
^^^^^

* Added an early-termination penalty and bounded the foot-contact reward so falls and
  hard impacts are not learned as high-reward behavior.  Existing checkpoints should be
  retrained because the reward scale changed.
