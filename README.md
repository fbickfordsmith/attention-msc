# When is visual attention useful?
MSc research by Freddie Bickford Smith\
Supervisors: Brad Love, Brett Roads, Ed Grefenstette\
University College London, September 2019

## Abstract
Attention, the ability to focus on relevant information, is known to aid human visual perception. But the cognitive-science literature lacks a systematic characterisation of how the impact of attention varies with the nature of a visual task. Happily, recent work has shown that deep convolutional neural networks are state-of-the-art models of the human visual system, meaning we can use them to conduct instructive large-scale studies. By training and evaluating more than 90 attention-augmented networks, we test the hypothesis that a visual task’s difficulty, size and perceptual similarity affect the usefulness of attention (the performance improvement that attention produces). Each task we consider is defined by a category set (a group of image categories); learning to apply attention to a particular category set represents a distinct cognitive task. We show that usefulness correlates positively and strongly (β_1=0.30, R^2=0.92) with category-set difficulty, negatively and strongly (β_1=-0.04, R^2=0.94) with category-set size (on a logarithmic scale), and negatively and weakly (β_1=-0.11, R^2=0.37) with the visual similarity within a category set. The first two relationships agree with our intuitions, but the third does not (we expected a positive correlation). These findings serve to inform not only basic research in cognitive science but also practical applications of visual attention in deep-learning systems.

>![](/data/figures/regression.png)
>Accuracy change produced by attention on 25 difficulty-based task sets (left), 20 size-based task sets (middle) and 40 similarity-based task sets (right). Task-set size is transformed logarithmically with base 2. Least-squares linear regression is applied to each subset of results; predictions of the linear models are shown as green lines.

## A brief guide to reproducing our results
Run the files in the table below. Then run the notebooks in `analysis/` to produce the plots and statistics presented in the paper. The code in this repository should work with TensorFlow v2.1.

\# | Step | Folder | File
-|-|-|-
1 | Compute mean accuracy of VGG16 on each ImageNet category | `experiments/` | `vgg16_testing.py`
2 | Compute mean VGG16 representation of each ImageNet category | `similarity/` | `representations_all.py` `representations_mean.py`
3 | Define category sets | `experiments/` | `define_cat_sets_difficulty.py` `define_cat_sets_semantic.py` `define_cat_sets_size.py` `define_cat_sets_similarity.py`
4 | Make dataframes listing examples for each category set | `experiments/` | `make_dataframes.py`
5 | Check attention works as intended | `experiments/` | `attention_networks_check.py`
6 | Train attention networks | `experiments/` | `attention_networks_training.py`
7 | Test attention networks | `experiments/` | `attention_networks_testing.py`
8 | Combine results from testing attention networks | `experiments/` | `combine_results.py`
