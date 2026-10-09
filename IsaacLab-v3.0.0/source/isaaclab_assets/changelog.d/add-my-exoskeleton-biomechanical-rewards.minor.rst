Changed
^^^^^^^

* Replaced the my_exoskeleton joint-motion momentum bonus with a centroidal angular-momentum
  cost, added lower-limb power and symmetry terms adapted from the paper, and made straight-knee
  shaping stance-aware. Existing checkpoints should be retrained because the reward objective changed.

Fixed
^^^^^

* Prevented a stationary double-support policy from receiving walking contact credit, added a
  zero-at-rest phase-one speed objective, and tightened the velocity-success threshold.
