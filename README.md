# When is visual attention useful?
MSc research by Freddie Bickford-Smith  
Supervisors: Brad Love, Brett Roads, Ed Grefenstette  
University College London, September 2019

## Abstract
Attention, the ability to focus on relevant information, is known to aid human visual perception. But the cognitive-science literature lacks a systematic characterisation of how the impact of attention varies with the nature of a visual task. Happily, recent work has shown that deep convolutional neural networks are state-of-the-art models of the human visual system, meaning we can use them to conduct instructive large-scale studies. By training and evaluating more than 90 attention-augmented networks, we test the hypothesis that a visual task’s difficulty, size and perceptual similarity affect the usefulness of attention (the performance improvement that attention produces). Each task we consider is defined by a category set (a group of image categories); learning to apply attention to a particular category set represents a distinct cognitive task. We show that usefulness correlates positively and strongly (β_1=0.30, R^2=0.92) with category-set difficulty, negatively and strongly (β_1=-0.04, R^2=0.94) with category-set size (on a logarithmic scale), and negatively and weakly (β_1=-0.11, R^2=0.37) with the visual similarity within a category set. The first two relationships agree with our intuitions, but the third does not (we expected a positive correlation). These findings serve to inform not only basic research in cognitive science but also practical applications of visual attention in deep-learning systems.

## Repository guide
Note that any reference to a **context** in this repository relates to what we call a **category set** (grouping of ImageNet categories) in the thesis (`_thesis.pdf`). We changed the terminology for the write-up to improve clarity.

All `.py` files in the root directory contain a docstring describing what they do.

For each experiment we use
- `contexts_def_[type_context].py` to define a set of contexts
- `contexts_dataframes.py` to build dataframes defining the training set for each context
- `contexts_training.py` to train an attention network on each context
- `contexts_testing.py` to evaluate these networks

`.py` files with prefix `representations` relate to computing VGG16 representations of images; those with prefix `similarity` relate to computing similarity measures using the representations. The table below gives an overview of the rest of the repository.

|Directory|Contents|
|-|-|
|`contexts`|Contexts used in experiments|
|`metadata`|ImageNet metadata|
|`plotting`|Figures for the thesis; code for generating them|
|`representations`|VGG16 representations of images; similarity matrices based on them|
|`results`|Performance of trained attention networks|
|`training`|In-training performance of attention networks|

## Citation
Bickford-Smith (2019). When is visual attention useful? MSc thesis, University College London.
