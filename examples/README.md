# Examples

This repository currently ships two distributable examples that share one dataset.

## Available examples

- `distro/example_01_scatter_equirect.sh`  
  Pose Scatter layer + equirectangular map + curvature background (`--pose-layer scatter --map equirect`)

- `distro/example_02_trace.sh`  
  Pose Trace layer + "world like" map + radial background (`--pose-layer trace`)

## Quick start

From repository root:

```bash
cd examples/distro
# run example 01
bash example_01_scatter_equirect.sh

# run example 02
bash example_02_trace.sh




See examples/distro/README.md for full command details and expected outputs.

