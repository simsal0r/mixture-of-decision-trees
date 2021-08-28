# MoDT: Mixture of Decision Trees
An interpretable DT ensemble method based on the Mixture of Experts architecture. The implementation is inspired by a similar approach called Mixture of Expert Trees: https://arxiv.org/abs/1906.06717 .

The Expectation-Maximization (EM) training algorithm iteratively optimizes a set of decison trees and associated regions:\
![](examples/output/example_gate1.gif?raw=true "Animation of the training process.")

Decision area of the final gating function:\
![](examples/output/example_gate1.jpg?raw=true "Final gating function.")

Resulting decision trees:
| DT0 red region | DT1 green region  | DT2 blue region |
| ------------- | ------------- | ------------- |
| ![](examples/output/example_dt3.svg?raw=true)  | ![](examples/output/example_dt1.svg?raw=true)  | ![](examples/output/example_dt2.svg?raw=true)  |

## A more sophisticated example
MoDT is used with the steel plates faults dataset: https://archive.ics.uci.edu/ml/datasets/Steel+Plates+Faults

The visualizations are created with dtreeviz: https://github.com/parrt/dtreeviz

![](examples/output/cs_steel_d2_gate.jpg?raw=true "Gating function.")

### DT 0 (red):
<img src="examples/output/cs_steel_d2_t0.svg" width="400">

### DT 1 (green):
<img src="examples/output/cs_steel_d2_t1_s.svg" width="400">

### DT 2 (blue):
<img src="examples/output/cs_steel_d2_t2.svg" width="600">

