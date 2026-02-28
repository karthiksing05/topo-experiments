"""
Topography implemented with the same sparsity constraints as project with Zekun!

Two main constraints:
*   A batch-wide KL constraint to ensure the space gets used properly
*   A per-instance entropy constraint to ensure individual instances have low entropy

We'll be implementing these constraints on the downsampled cortical sheet of activations to
ensure that when scaling back up that sparsity is retained per-cluster and not per-neuron
(although honestly we can try all sorts of stuff).

The goal is to be able to visualize activations and have specific clusters not just be selective,
but also light up for a given task.
"""

