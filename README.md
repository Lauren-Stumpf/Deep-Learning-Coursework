# Generative Modelling: Generating realistic images with computational limitations
Submitted as part of the degree of Msci Natural Sciences (3rd year) to the Board of Examiners in the Department of Computer Sciences, Durham University. 
This summative assignment was assessed and marked by the professor of the module in question:
## Grade: 1st - 100/100 (100 + bonus of 4), 1st in year (of 136 students).
> "This is a perfect GAN-based submission, where you’ve clearly done a significant amount of research by
> reading of state-of-the-art peer-reviewed literature from top venues (CVPR, ICCV, NeurIPS). The samples
> have excellent diversity with very good coverage and few issues, usually only noticable in the backgrounds
> at the sides. The model samples are amongst the best in the class - I’ve used several of them as
> exemplars in the general feedback. The interpolations clearly show the model captures the data distribution, with a smooth and
> intuitive generated images between a diverse set of samples." - Dr Chris G. Willcocks

## Paper Abstract:
After successfully implementing and experimenting with two flow-based generative models [1], namely Glow [6] and the multi-scale augmented normalizing flow proposed in [9], and various variational autoencoder (VAE)
[5] architectures, the study moved on to Generative Adversarial Networks. This paper proposes using a StyleGAN2 architecture in combination with an adaptive discriminator augmentation (ADA) mechanism trained on the
FFHQ dataset. This mechanism passes the images shown to the discriminator through an augmentation pipeline and dynamically adjusts the strength of augmentation. We produce realistic interpolation and achieve a reasonable balance between diversity and quality of images

All models were implemented with the goal of generating diverse and realistic images from the FFHQ dataset given severe computational limitations (access to one NVIDIA TITAN Xp or an NVIDIA GeForce RTX 2080 for a maximum of three training days).

## Contents:
* deep_learning_paper.pdf - Paper reporting findings and methodology of the study, "Generating Images using StyleGAN2 + ADA".
* source-code/ - Folder containing the herein modified version of lucidrains' implementation of state-of-the-art StyleGAN2 + ADA proposed in ICLR 2021 (this source-code has yet to be cleaned and commented, apologies). Credits for paper and code go to:
    - StyleGAN2 + ADA paper - [Training Generative Adversarial Networks with Limited Data](https://arxiv.org/pdf/2006.06676.pdf).
    - [lucidrains](https://github.com/NVlabs/stylegan2-ada-pytorch/)' original implementation - https://github.com/NVlabs/stylegan2-ada-pytorch.

## Results (taken from pegasus-paper.pdf):
![Best Pegasus](results-image-1.png?raw=true "Best Pegasus")
![Other Pegasi](results-image-2.png?raw=true "Other Pegasi")
