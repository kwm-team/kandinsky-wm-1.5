# AR autoregressive video

AR generates the next video chunk from its startup context or generated history and the next camera poses. The native rollout maintains its causal state across chunks, including the solver schedule and cache policy. Use a standalone trajectory covering the entire requested rollout. [Coordinate convention and example inputs](../examples/README.md)

The command-line renderer writes an MP4 from fixed inputs. Applications can use the incremental runtime to supply new poses at each generation call. On one H100 at 768×512, the eager DiT took about 1.5–1.8 seconds per 16-frame chunk. This measurement covers transformer generation. The VAE, output encoding and transport contribute additional latency.

The final-step GAN is a distinct optional component. Example renders and timings identify whether that component was used.
