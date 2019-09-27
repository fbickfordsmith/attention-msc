# When is visual attention useful?
MSc research by Freddie Bickford-Smith

Supervisors: Brad Love, Brett Roads, Ed Grefenstette

University College London, September 2019

## Abstract
Visual attention, the ability to prioritise the processing of selected visual information, is key to how we perceive the world. Recent work has shown that deep convolutional neural networks are good models of human vision. We use such a network, incorporating attention as a multiplicative weighting of neural representations of visual stimuli. We define a category set to be a grouping of ImageNet categories. Learning to apply attention to a particular category set represents a distinct cognitive task. We hypothesise that a category set's difficulty, size and visual similarity influence the usefulness of attention (the accuracy improvement attention provides on a task). Using results from more than 90 trained computer models, we show that usefulness correlates positively ($\beta_1=0.30$) and strongly ($R^2=0.92$) with category-set difficulty, negatively ($\beta_1=-0.04$) and strongly ($R^2=0.94$) with category-set size (on a logarithmic scale), and negatively ($\beta_1=-0.11$) and weakly ($R^2=0.37$) with the visual similarity within a category set. The first two relationships agree with our intuitions; the third does not (we expected a positive correlation). Future efforts could focus on building more sophisticated models of visual attention. But there are, we argue, a number of natural extensions to our work that offer more tractable steps towards a better computational account of visual attention.

## Repository guide
Note that any reference to a **context** in this repository relates to what we call a **category set** (grouping of ImageNet categories) in the thesis (`_thesis.pdf`). We changed the terminology for the write-up to improve clarity.

Most `.py` files in the root directory contain a docstring describing what they do. The table below gives an overview of the rest of the repository.

|Directory|Contents|
|-|-|
|`contexts`|Definitions of category sets|
|`metadata`|ImageNet metadata|
|`plotting`|Figures for the thesis; code for generating them|
|`representations`|VGG16 representations of images; similarity matrices based on them|
|`results`|Performance of trained attention networks|
|`training`|In-training performance of attention networks|
