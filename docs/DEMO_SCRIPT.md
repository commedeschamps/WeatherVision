# WeatherVision presentation

**Author:** Rakhman Seidigali  
**Group:** SE-2411  
**Target duration:** approximately four minutes

## Presentation script

**0:00-0:30 - Introduction**

Hello. I am Rakhman Seidigali from group SE-2411. Our Computer Vision project is WeatherVision. The application classifies photographs into four categories: Cloudy, Rain, Shine, and Sunrise. We compare a small CNN trained from scratch with two pretrained architectures, MobileNetV2 and ConvNeXt-Tiny, and present the results in an interactive web application.

**0:30-1:10 - Dataset and preprocessing**

We use the Multi-class Weather Dataset for Image Classification published by Gbeminiyi Ajayi. We obtained the images through the PerceptiLabs mirror. The downloaded files contained 1,123 valid weather photographs. After removing 26 exact duplicates, we retained 1,097 images.

Before splitting, we grouped detected near duplicates so that related images stay in the same partition. The final dataset contains 768 training, 164 validation, and 165 test images. All experiments use this same membership and seed 42. Images are resized, center cropped to 224 by 224, and normalized. Horizontal flips, small rotations, and color jitter are applied during training.

**1:10-1:55 - Models and results**

The Simple CNN has three convolutional blocks followed by global pooling and a four-class classifier. MobileNetV2 and ConvNeXt-Tiny start with ImageNet-pretrained features. We freeze their feature extractors and train a new four-class linear head.

We use cross-entropy loss and AdamW. Validation macro-F1 selects the best checkpoint. On the shared test set, Simple CNN achieved 95.15 percent accuracy, MobileNetV2 achieved 96.36 percent, and ConvNeXt with augmentation achieved 98.18 percent.

For the ablation, we trained ConvNeXt without augmentation while keeping the other settings fixed. Its test accuracy was 97.58 percent. The augmented model classified one additional test image correctly, and macro-F1 increased from 0.9759 to 0.9827.

**1:55-2:55 - Application demonstration**

Here is the WeatherVision application. I choose ConvNeXt-Tiny, select a test photograph, and press Classify image. The app loads the saved checkpoint and displays the predicted category, all four softmax scores, the forward-pass time, and the device. We can also upload another image or choose one of the other trained models. CPU inference is supported.

Below the prediction, Grad-CAM shows an approximate spatial explanation. The original photograph appears beside a heatmap overlay. Brighter regions indicate features associated with the selected class score. The implementation maps the network's center crop back onto the original image. We saved examples of both correct and incorrect predictions to help analyze model behavior.

**2:55-3:35 - Comparison and reproducibility**

The Experiments tab contains the measured accuracy, precision, recall, and macro-F1 for every experiment. It also displays training curves and confusion matrices. Macro-F1 gives equal weight to each category, and the confusion matrix shows which labels were confused.

The latency benchmark uses five warm-up runs and 30 synchronized batch-one forward measurements. We retain the raw measurements together with the hardware information. The dataset manifest, configurations, checkpoints, and predictions are also saved. A Colab notebook runs the same workflow and supports training resume and output export.

**3:35-4:05 - Conclusion**

ConvNeXt with augmentation achieved the highest test accuracy in our comparison. MobileNetV2 improved on the CNN baseline using pretrained visual features, while the small CNN provided the fastest measured forward pass. The project brings together reproducible data preparation, transfer learning, controlled ablation, visual explanations, and a working application. Further development can extend the dataset across more locations, seasons, and cameras. Thank you.

## Recording storyboard

| Time | Screen | Action |
|---|---|---|
| 0:00-0:30 | App title | Introduce the team and four classes. |
| 0:30-1:10 | Dataset inspection and distribution | Explain counts, duplicate grouping, and the split. |
| 1:10-1:55 | Comparison table | Describe the models and read the measured results. |
| 1:55-2:20 | Classify tab | Select ConvNeXt, load an image, and classify it. |
| 2:20-2:55 | Prediction and Grad-CAM | Explain class scores and spatial attribution. |
| 2:55-3:35 | Experiments, curves, confusion matrix | Explain metrics, timing, and reproducibility. |
| 3:35-4:05 | Report and app | State the conclusion and future development. |

## Preparation

1. Run `python -m streamlit run app.py` and check the saved checkpoints.
2. Prepare an example image and a saved Grad-CAM figure.
3. Record the screen with audible narration at a readable zoom level.
4. Review playback for image quality, sound, and duration.
5. Add the repository and video URLs to `configs/submission.yaml` when available, then regenerate the reports.
