# WeatherVision defense guide

**Author:** Rakhman Seidigali  
**Group:** SE-2411

The following explanations and answers describe the implemented models and the measured results in `outputs/model_comparison.csv`.

## Concepts in plain English

**Image classification:** assigning one category to an entire image. Here the four classes are Cloudy, Rain, Shine, and Sunrise. The output is one label and four class scores.

**Dataset:** a collection of examples and their labels. A label tells the learning algorithm which category the example belongs to. Our images come from Ajayi's public dataset, through a resized PerceptiLabs mirror.

**Training, validation, test:** the model learns weights from training images. Validation images help choose the best checkpoint and stopping point. Test images are reserved for the final comparison. All models have the same membership. The manifest records it.

**CNN:** a convolutional neural network learns spatial patterns such as edges, textures, clouds, and combinations of features.

**Convolution:** a small filter moves across a feature map. Its shared weights compute a local response at each position, reducing the number of weights compared with connecting every pixel to every neuron.

**Pooling:** reducing spatial resolution by summarizing a region. Our Simple CNN uses max pooling and final global average pooling. This reduces computation while retaining useful features, but loses fine spatial detail.

**Batch normalization:** normalizing intermediate training activations with learned scale and offset. Frozen MobileNet feature layers use their pretrained running statistics rather than changing them on this small dataset.

**Transfer learning:** reusing features learned on a large source dataset, then adapting the model to a new task. Here only the new four-class linear head trains. We did not fine-tune the feature extractors.

**MobileNetV2:** an efficient CNN with depthwise convolutions, inverted residual blocks, and linear bottlenecks. It offers useful pretrained image features with fewer parameters than ConvNeXt.

**ConvNeXt:** a modern CNN family using hierarchical stages, large depthwise kernels, and other design changes. The Tiny variant is the advanced architecture evaluated in Assignment 2.

**Pretrained weights:** numerical parameters already learned on another dataset, in this case ImageNet-1K. They can reduce the amount of new labeled data needed. ImageNet accuracy is not our weather accuracy.

**Overfitting:** learning the training examples too specifically, so performance on unseen data suffers. A widening train/validation gap is a possible warning, not a complete diagnosis.

**Data augmentation:** creating random training views, such as a horizontal flip or a small rotation. Labels stay the same. We also use mild color jitter. Color affects weather scenes, so this choice must be evaluated rather than assumed useful. Validation and test images are never randomly augmented.

**Ablation study:** a controlled comparison that changes one factor. We compare ConvNeXt with and without augmentation, keeping the other configuration fields and split identical.

**Grad-CAM:** gradients of one class score weight the final spatial feature maps. Positive weighted responses produce a coarse heatmap. Brighter areas indicate approximate spatial feature attribution for that class. The original image is retained, and the center crop is mapped back correctly.

**Accuracy:** correct predictions divided by all predictions. It is easy to understand but can hide poor performance on a smaller class.

**Precision:** among images predicted as a class, how many really belong to it? For Rain, this measures false alarms.

**Recall:** among images actually in a class, how many were found? For Rain, this measures missed rainy scenes.

**F1:** the harmonic mean of precision and recall: `2PR/(P+R)`. It becomes low when either precision or recall is low. Macro-F1 averages the four class F1 scores equally.

**Confusion matrix:** a table with true labels in rows and predicted labels in columns. The diagonal contains correct predictions. Other cells reveal which categories are confused.

**Small dataset:** roughly a thousand images provide limited coverage of cameras, locations, lighting, seasons, and mixed scenes. Duplicates reduce independent information further. A large pretrained network has many more weights than this dataset has examples, so we freeze its backbone.

**Visual categories:** Cloudy emphasizes cloud texture; Rain can include rain streaks and wet surfaces; Shine emphasizes bright sunlight; Sunrise contains characteristic warm horizon lighting. The classifier assigns one of these dataset labels to each photograph.

## Realistic questions and concise answers

1. **What problem do you solve?** Four-way classification of outdoor photographs using CNNs and transfer learning.
2. **What does the application return?** The predicted category, four softmax scores, inference time, device information, and an optional Grad-CAM explanation.
3. **Why is Sunrise included?** The dataset contains a Sunrise class with characteristic warm lighting and horizon patterns.
4. **Which dataset do you use?** The Multi-class Weather Dataset for Image Classification by Gbeminiyi Ajayi, published in 2018 and distributed under CC BY 4.0.
5. **How do Assignments 1 and 2 connect?** Assignment 1 compares Simple CNN and MobileNetV2. Assignment 2 extends the same pipeline with ConvNeXt-Tiny, augmentation ablation, and Grad-CAM.
6. **Why not quote exactly 1,125 images?** We counted actual valid files. The mirror has 1,123 valid weather images before deduplication. Published totals and mirrors can differ.
7. **How many images did you retain?** The current audit retained 1,097 after removing 26 exact duplicates. The manifest has 768 train, 164 validation, and 165 test images.
8. **How do you handle duplicates?** File SHA-256 and decoded-pixel hashes detect exact copies. Conservative perceptual/difference hashes, aspect ratio, and thumbnail RMSE detect near duplicates. Connected groups are formed before splitting.
9. **How are near-duplicate images handled?** Images meeting the combined pHash, dHash, aspect-ratio, and thumbnail-color thresholds are grouped before splitting. Candidate pairs are recorded in a CSV.
10. **What happens to conflicting duplicate labels?** The entire connected component is quarantined rather than assigning an arbitrary label.
11. **Why 70/15/15?** It gives most data to learning while retaining validation and untouched test examples. Whole duplicate groups make the proportions approximate.
12. **Why seed 42?** A fixed conventional seed makes splits and initialization reproducible. The value itself has no scientific advantage.
13. **Can two devices give different results?** Yes. Backend operations and software versions can differ. Hardware, dependencies, and deterministic settings are recorded with each experiment.
14. **Why normalize with ImageNet statistics?** The transfer backbones expect that input scale. The same normalization is applied to the baseline for a clear protocol.
15. **Why different resize sizes?** The official V1 pretrained transforms use shorter-side resize 256 for MobileNetV2 and 236 for ConvNeXt, followed by a 224 center crop.
16. **What is the baseline architecture?** Three 3x3 convolution blocks with 16, 32, and 64 channels, batch normalization, ReLU and pooling, then global pooling, dropout, and a four-output linear layer.
17. **Why use cross-entropy?** This is mutually exclusive four-class classification. The model returns raw logits, and cross-entropy combines log-softmax and the target-label loss.
18. **Why AdamW?** It is a practical adaptive optimizer with decoupled weight decay. We use a fixed documented learning rate rather than claiming it is optimal.
19. **Which parameters train in transfer learning?** Only the new four-output linear head. MobileNet has 5,124 trainable parameters and ConvNeXt has 3,076 in this implementation.
20. **Why keep frozen layers in evaluation mode?** Otherwise MobileNet batch-normalization statistics or ConvNeXt stochastic depth could still change the feature behavior despite frozen weights.
21. **How is the best model selected?** Highest validation macro-F1; ties keep the earliest checkpoint. Test metrics never choose the checkpoint.
22. **What is early stopping?** Stop after four epochs without a strictly better validation macro-F1. All current full runs reached their configured epoch limit.
23. **What distinguishes ConvNeXt?** It is a modern CNN using hierarchical stages, large depthwise kernels, LayerNorm, and an inverted bottleneck structure.
24. **Why not fine-tune the whole backbone?** The dataset is small. Frozen features give a manageable controlled foundation. The project trains a compact linear head on top of the frozen representations.
25. **What changed in your ablation?** Only training augmentation. Architecture, weights identifier, initialization protocol, seed, shuffle, optimizer, learning rate, batch size, maximum epochs, and stopping rule were fixed.
26. **What did the augmentation ablation show?** Test accuracy increased from 97.58% to 98.18%, equivalent to one extra correct prediction among 165 test images. Macro-F1 increased from 0.9759 to 0.9827.
27. **Why macro-F1 for model selection?** It gives equal importance to each category and balances precision and recall. Class counts are unequal.
28. **Why is test accuracy higher than validation accuracy for the CNN?** These are different finite samples, and difficulty can vary. The split is small. Detected groups and file hashes are verified before evaluation.
29. **What is softmax confidence?** A normalized model score among the four categories. The four scores sum to one.
30. **Can the model reject a picture of a cat?** No explicit rejection class or detector was trained. It will still assign one of the four labels, possibly with high confidence.
31. **How do you compute latency?** Batch-one forward passes on a device-resident test tensor, after five warm-ups, repeated 30 times with GPU synchronization. We report the arithmetic mean and median and retain raw timings.
32. **Why can a smaller model appear slower?** Kernel implementation, device scheduling, and architecture affect latency. Parameter count alone is not a latency prediction.
33. **Does latency include loading or preprocessing?** The comparison benchmark excludes both, transfer, and softmax. The app's single-pass timer includes softmax but excludes preprocessing and loading, and is separately labeled.
34. **How do gradients work when the backbone is frozen?** Grad-CAM enables gradients on a cloned input and requests the gradient of the class logit with respect to the final spatial activation. No optimizer update occurs.
35. **Which layer does Grad-CAM use?** ConvNeXt `features[-1]`, which retains spatial dimensions before pooling and the classifier.
36. **What does a bright heatmap region represent?** It shows spatial features with a positive contribution to the selected class score under the Grad-CAM calculation.
37. **Does the app train on upload?** No. It loads full saved checkpoints, caches models, and performs inference. Uploaded photos are not written to disk by the app.
38. **Can it run on CPU?** Yes. Tests cover CPU loading and predictions for all three models. The UI defaults to CPU and offers automatic accelerator selection.
39. **How do you reproduce the results?** Install dependencies, run `python scripts/run_all.py --resume`, and inspect the stored manifest/configuration/hardware metadata. Existing completed runs are checked and skipped.
40. **Who developed the project?** Rakhman Seidigali, group SE-2411, Astana IT University.

## Demonstration preparation

Run `python -m streamlit run app.py`. Open Classify, select a trained model, use a test photograph or upload an outdoor image, and press Classify image. Compare all models, show the shared test metrics and confusion matrix, then explain one correct and one incorrect Grad-CAM example. Keep `outputs/` checkpoints on the demonstration computer. Use `docs/DEMO_SCRIPT.md` for timing.

41. **Is this weather forecasting?** It classifies the visible appearance of one photograph. Forecasting predicts future weather and requires time-dependent observations and a different model.
42. **Why is ConvNeXt suitable as the advanced method?** Its 2022 design uses larger depthwise filters and a deeper feature hierarchy than the baseline. The pretrained features work well with a small four-class head.
43. **What does the latency average mean?** Add the 30 warmed forward times and divide by 30. The median is the middle value and is less sensitive to slower individual calls.
44. **Does browser choice change the inference device?** No. The browser sends the image to Streamlit; PyTorch runs on the selected server backend, such as CPU or MPS.
45. **Did you generate training photographs?** No. All training images come from the cited dataset. Augmentation changes their appearance during loading; it does not add generated source images. The demo uses synthetic speech, which is separate from training.
46. **How does an instructor obtain your model?** Clone the public repository and run scripts/download_checkpoints.py. It downloads release files and validates their SHA-256 checksums before inference.
47. **Was the notebook executed on hosted Colab?** It was executed with a local Python kernel. It contains the shared workflow and saved outputs; hosted Colab execution has not been claimed.
48. **Why is the ablation gain small?** Both runs begin with strong pretrained features. The test-set difference is one image, so additional locations and repeated seeds would help evaluate how consistent that difference is.
