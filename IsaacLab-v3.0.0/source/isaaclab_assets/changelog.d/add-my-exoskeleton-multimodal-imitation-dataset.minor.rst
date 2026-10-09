Multimodal exoskeleton imitation dataset loader
-------------------------------------------------

* Added task and IMU mapping YAML files and a time-synchronized
  ``ExoskeletonImitationDataset`` for the recorded BT/AT trials.
* Added optional lazy RGB and AEDAT4/event-stream decoding.  Event coordinates
  are calibrated from the supplied 320x240 sensor grid before histogramming;
  RGB frame differencing is never used as an event fallback.
* Trials without calibrated eight-joint motion files are marked
  ``motion_valid=False`` instead of silently treating IMU data as joint data.
