# Generative Modelling: Generating Realistic Images with Computational Limitations and Interpolating Between Them. 

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
This paper proposes using a StyleGAN2 architecture in combination with an adaptive discriminator augmentation (ADA) mechanism trained on the
FFHQ dataset. This mechanism passes the images shown to the discriminator through an augmentation pipeline and dynamically adjusts the strength of augmentation. We produce realistic interpolation and achieve a reasonable balance between diversity and quality of images

All models were implemented with the goal of generating diverse and realistic images from the FFHQ dataset given severe computational limitations (access to one NVIDIA TITAN Xp or an NVIDIA GeForce RTX 2080 for a maximum of three training days).

## Contents:
* styleGAN2_paper.pdf - Paper reporting findings and methodology of the study, "Generating Images using StyleGAN2 + ADA".
* source_code/ - Folder containing the modified version of te state-of-the-art StyleGAN2 + ADA proposed in ICLR 2021 (this source-code has yet to be cleaned and commented, apologies). I have mainly condensed the code and have made it more suitable for a single GPU as per the assignment. Credits for paper and code go to:
    - StyleGAN2 + ADA paper - [Training Generative Adversarial Networks with Limited Data](https://arxiv.org/pdf/2006.06676.pdf).
    - [lucidrains](https://github.com/NVlabs/stylegan2-ada-pytorch/)' original implementation - https://github.com/NVlabs/stylegan2-ada-pytorch.

## Results (taken from pegasus-paper.pdf):
* Best Images Generated
![Generated Images 1](images/complete_1.png?raw=true "Generated Images 1")
![Generated Images 2](images/complete_2.png?raw=true "Generated Images 2")
![Generated Images 3](images/complete_3.png?raw=true "Generated Images 3")
* Random Sample of Images Generated
![Generated Final](images/final.png?raw=true "Final")
* Interpolated Images 
    * This is done by linearly interpolating between points in the w latent space 
![Interpolated Images 1](images/inter_1.png?raw=true "Interpolated 1")
![Interpolated Images 2](images/inter_2.png?raw=true "Interpolated 2")
![Interpolated Images 3](images/inter_3.png?raw=true "Interpolated 3")
![Interpolated Images 4](images/inter_4.png?raw=true "Interpolated 4")
![Interpolated Images 5](images/inter_5.png?raw=true "Interpolated 5")
![Interpolated Images 6](images/inter_6.png?raw=true "Interpolated 6")
![Interpolated Images 7](images/inter_7.png?raw=true "Interpolated 7")


