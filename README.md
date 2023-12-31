# RotDiff

This is the source code for paper RotDiff in CIKM'23.

All code is based on Python and PyTorch.

Research problem: Information Diffusion Prediction

Paper Title: RotDiff: A Hyperbolic Rotation Representation Model for Information Diffusion Prediction

Abstract: The massive amounts of online user behavior data on social networks allow for the investigation of information diffusion prediction, which is essential to comprehend how information propagates among users. The main difficulty in diffusion prediction problem is to effectively model the complex social factors in social networks and diffusion cascades. However, existing methods are mainly based on Euclidean space, which cannot well preserve the underlying hierarchical structures that could better reflect the strength of user influence. Meanwhile, existing methods cannot accurately model the obvious asymmetric features of the diffusion process. To alleviate these limitations, we utilize rotation transformation in the hyperbolic to model complex diffusion patterns. The modulus of representations in the hyperbolic space could effectively describe the strength of the user's influence. Rotation transformations could represent a variety of complex asymmetric features. Further, rotation transformation could model various social factors without changing the strength of influence. In this paper, we propose a novel hyperbolic rotation representation model RotDiff for the diffusion prediction problem. Specifically, we first map each social user to a Lorentzian vector and use two groups of transformations to encode global social factors in the social graph and the diffusion graph. Then, we combine attention mechanism in the hyperbolic space with extra rotation transformations to capture local diffusion dependencies within a given cascade. Experimental results on five real-world datasets demonstrate that the proposed model RotDiff outperforms various state-of-the-art diffusion prediction models.

Please cite us if you use our code. Thanks!

```latex
@inproceedings{Qiao2023,
    author = {Qiao, Hongliang and Feng, Shanshan and Li, Xutao and Lin, Huiwei and Hu, Han and Wei, Wei and Ye, Yunming},
    title = {RotDiff: A Hyperbolic Rotation Representation Model for Information Diffusion Prediction},
    year = {2023},
    address = {New York, NY, USA},
    doi = {10.1145/3583780.3615041},
    pages = {2065–2074},
    numpages = {10},
    location = {Birmingham, United Kingdom},
    series = {CIKM '23}
}
```

