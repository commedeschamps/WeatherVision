# WeatherVision references

Primary-source metadata checked November 2, 2026.

| Source | Metadata | Primary reference |
|---|---|---|
| Weather dataset | Gbeminiyi Ajayi; 13 September 2018; Mendeley Data V1; CC BY 4.0 | https://data.mendeley.com/datasets/4drtyfjtfy/1 |
| PerceptiLabs mirror | cloud/rain/shine/sunrise directories; resized images; Ajayi attribution | https://github.com/PerceptiLabs/Weather-Analysis |
| MobileNetV2 | Sandler, Howard, Zhu, Zhmoginov, Chen; CVPR 2018; pp. 4510-4520 | https://arxiv.org/abs/1801.04381 |
| ConvNeXt | Liu, Mao, Wu, Feichtenhofer, Darrell, Xie; A ConvNet for the 2020s; CVPR 2022; arXiv v2 | https://arxiv.org/abs/2201.03545v2 |
| Grad-CAM | Selvaraju, Cogswell, Das, Vedantam, Parikh, Batra; arXiv v3, 21 March 2017 | https://arxiv.org/abs/1610.02391v3 |
| ConvNeXt preprocessing | ImageNet V1 shorter-side resize 236, center crop 224, ImageNet normalization | https://docs.pytorch.org/vision/master/models/generated/torchvision.models.convnext_tiny.html |
| MobileNetV2 implementation | ImageNet V1 weights and transform preset | https://docs.pytorch.org/vision/main/_modules/torchvision/models/mobilenetv2.html |

The pinned mirror commit and downloaded archive SHA-256 are recorded in `data/source.json`. Image counts come from the local file audit in `data/statistics.json`. BibTeX records are available in `docs/references.bib`.


## Final model citation check — 2026-11-03

ConvNeXt is a 2022 architecture. The original Liu et al. paper and official TorchVision 0.25 implementation support the four-stage design, depthwise 7x7 blocks, LayerNorm/GELU, V1 weight selection and preprocessing. The implemented head is Linear(768, 4). It is presented as a modern CNN, with no claim of being the newest or globally best architecture.

Queries `site:neurohive.io "ConvNeXt"` and `site:neurohive.io "A ConvNet for the 2020s"` returned no applicable Neurohive page. The report cites the original paper and official implementation instead. No Neurohive reference was fabricated.

- [Liu et al., A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
- [TorchVision 0.25 ConvNeXt source](https://docs.pytorch.org/vision/0.25/_modules/torchvision/models/convnext.html)

The synthetic training-image share is 0%: all 1,097 retained manifest entries map to validated photographs from the pinned archive. Preparation has no image-generation step. Training augmentation transforms loaded source photographs; tests use separate temporary fixtures.
