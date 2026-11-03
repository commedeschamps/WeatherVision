# WeatherVision final demonstration script

Rakhman Seidigali · SE-2411 · Computer Vision · Astana IT University

Duration: 4:08.

The published 4:08 video records the actual Streamlit application and uses macOS Samantha synthetic English narration, with captions in `WeatherVision_demo.srt`. The recording is available from the release link in the README; the local MP4 is retained in `dist/`.

## 0:00–0:31 — Introduction

Hello. My name is Rakhman Seidigali, from group S E twenty four eleven at Astana I T University. This is WeatherVision, my Computer Vision project. It classifies outdoor photographs into four categories: Cloudy, Rain, Shine, and Sunrise. The application brings together three trained models, a comparison of their results, and visual explanations. This demonstration shows the running application and its saved experimental results.

## 0:31–1:21 — Dataset and models

The project uses the Multi-class Weather Dataset published by Gbeminiyi Ajayi. After duplicate removal, it contains one thousand and ninety seven photographs. There are seven hundred and sixty eight training images, one hundred and sixty four validation images, and one hundred and sixty five test images. All experiments use the same split. The Simple C N N learns three convolutional blocks from scratch. MobileNet version two reuses efficient ImageNet features. ConvNext Tiny uses deeper convolutional stages and larger spatial filters. For both pretrained networks, only the new four class classification head is trained. Training augmentation includes horizontal flips, small rotations, and mild color changes.

## 1:21–2:51 — Live application

I now choose Upload and select a weather photograph. The preview lets me check the input before prediction. I select ConvNext Tiny and press Classify image. The application loads its saved checkpoint and displays the predicted category. The confidence is the model's softmax score, and the four bars show how the scores are distributed across the labels. The inference backend here is C P U. The timing panel shows a single forward pass and the average of thirty measurements after five warm up runs. Model loading and image preparation are excluded from this benchmark. The environment panel identifies the software and the browser. Next, Compare on this image runs all three model families on the same input. The comparison can be downloaded as a C S V file. For ConvNext, Show Grad Cam adds an explanation. The original image appears beside the heatmap overlay. Brighter regions indicate spatial features associated with the predicted class. This is an approximate attribution, rather than a precise segmentation of clouds or rain. I can download the class scores as J S O N and save the overlay as a P N G image. Switching the model keeps valid results for the same photograph.

## 2:51–3:37 — Experimental results

The Experiments tab compares all four training configurations on the same one hundred and sixty five test images. Simple C N N reaches ninety five point one five percent accuracy. MobileNet reaches ninety six point three six percent. ConvNext with augmentation reaches ninety eight point one eight percent, with macro F one of zero point nine eight two seven. Without augmentation, ConvNext reaches ninety seven point five eight percent. The difference is one additional correctly classified image: one hundred and sixty two instead of one hundred and sixty one. The learning curves show training and validation behavior. The confusion matrix shows true labels in rows and predictions in columns.

## 3:37–4:08 — Conclusion and access

ConvNext with augmentation has the highest accuracy in this comparison. The small C N N provides the shortest measured forward time on Apple M P S. The test set is small, and ambiguous labels or unusual lighting can affect predictions in new environments. The Project tab contains dataset attribution and both reports. Source code, configurations, an executed notebook, and downloadable model weights are available in the public GitHub repository. Thank you for watching.
