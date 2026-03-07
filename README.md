# Topography experiments for Karthik

A number of topography-related things I want to test and mess around with - I'll leave my work here to see if any intuitions can be extrapolated! It'll be largely interp-focused with some training experiments!

The overarching thesis here is that topography necessitates some performance upgrade - if we are building topographic models to have categorical selectivity, we should attempt to leverage that categorical selectivity on a per-instance basis for efficiency gains.

# Topographic Sparsity

Applying the batch-kl and per-instance-entropy penalty to every DOWNSAMPLED layer that we apply topoloss to in order to see effects!

Notes so far:
*   Keep temperature for softmax at 1, that's pretty chill
*   Entropy penalty does most of the work - small KL penalty to encourage the whole cortical sheet being explored

Note that there are other ways to induce sparsity!
*   Removing 

Why Topographic Sparsity? Most likely to induce better downstream performance for finetuning performance!! At the very least, catastrophic forgetting!!
*   By inducing sparsity in regions, we're inducing sparsity in gradients, which will result in sparse representation and robustness of representations!
*   Useful versus useless polysemanticity!!

# Different way to induce topography?

Immediately, my brain goes to some form of dynamic allocation - we shouldn't be forcing neurons to fit in a box, we should think of topography as allocation. The idea with Topoloss is that we're essentially partitioning the cortical sheet into a grid and trying to make neurons within the box fit with other neurons in the box - however, this says nothing about two given boxes and their similarities

Additionally, Topography says nothing about forcing the nature of the neurons into similarity - it's more of a dynamic allocation strategy that just works out to happening that way, where neurons next to other neurons can influence activity and the results are that we just naturally get topographic boundaries

I want to create my own topography-induced penalty, and make the feature-selectivity of neurons related to the need for input data (almost like an allocation strategy that's inherently topographic)
*   Something like a stem-cell philosophy? Where areas are able to shift and change to focus more on certain stimuli
*   Instead of trying to create one-size-fits-all neurons, we create neurons that develop selectivity within the architecture

*   Idea #1 - batches of fitting!!! "neuroplasticity-type fitting"
    *   Instead of training all neurons at the same time, we train sets of neurons, dynamically allocating additional neurons based on some error threshold after a given amount of epochs. The idea is that if a given 
    *   We can make it progress in terms of layers having pyramids with X selective areas that gradually develop (that way, later layer high-level computation increases slower than early-level computation and we can grow specific complexities)
    *   Each of the selective areas should also be sparse (i.e. if an area develops selectivity for faces, it should be the primary factor in faces) - backprop will guarantee that large corrections be given 
        *   If large corrections are given a lot, we expand the area topographically!!

*   NOTE: discussed this a TON with Claude, feeling OK about this at the high-level!! Not sure on other stuff but it might be interesting!