# CASTLE

### Modifications performed on original CASTLE code
- sigmoid activation added to the output layer of the sub-network which predicts our binary target
- binary cross entropy loss used as prediction loss
- restructured model initialization, required to be able to apply our model training pipeline

### References
Kyono, T., Zhang, Y., & van der Schaar, M. (2020). CASTLE: regularization via auxiliary causal graph discovery. Advances in Neural Information Processing Systems, 33, 1501-1512. ([GitHub](https://github.com/trentkyono/CASTLE "CASTLE repository"))
