# UCPE camera conditioning

UCPE uses the input image to establish appearance and the submitted camera path to control the observing viewpoint. Describe the surroundings and other road users in the text prompt. To compare camera paths, hold the prompt, first frame and seed fixed.

Submit a standalone trajectory with poses and pixel intrinsics. For the shared example, the first pose is the world origin and the first pose's image matches the conditioning frame. Camera axes are OpenCV right, down and forward. The quickstart includes a straight trajectory and a left turn. [Coordinate convention and example inputs](../examples/README.md)

The release checkpoint combines the exact source dense model and the learned camera adapter. The runtime uses the merged inference weights. The model-specific documentation gives the input format, launch command and numerical comparison procedure.
