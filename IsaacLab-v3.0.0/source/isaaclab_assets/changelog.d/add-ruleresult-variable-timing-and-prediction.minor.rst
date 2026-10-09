Added
^^^^^

* Extended the exoskeleton ``ruleresult`` playback output with simulation, configured, and
  monotonic wall-clock timing metadata. Derivatives now use the validated sample interval,
  while preserving the eight-dimensional action/checkpoint contract.
* Added optional constant-acceleration extrapolation and a supplied future-target callback for
  exporting labeled multi-step angle, velocity, and angular-acceleration sequences. The
  existing actor remains a one-step policy unless a separately trained multi-step head or
  rollout provider is supplied.
