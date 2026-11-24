# Assignment 1: Convolutional Neural Networks

## Overview

CIFAR-10 contains 50,000 training images and 10,000 test images across 10 classes. Each image is a small 32 × 32 colour picture. The training data was divided into training and validation sets, while the official test set was kept separate.

The experiments covered normalization, kernel sizes, pooling, network depth, dropout, weight decay, and data augmentation using a basic CNN. Each recorded comparison used two epochs and a 7,200-image training subset with an 800-image validation subset. These short runs are useful for comparing settings, but they are preliminary; they are not results from fully training every model on all 50,000 training images.

## Q1 — Loading and preprocessing

Ten CIFAR-10 sample images were displayed with their class names. Pixel values scaled to [0, 1] were compared with channel standardization, which subtracts the CIFAR-10 channel mean and divides by its standard deviation. Validation and test images used the same normalization as the corresponding training run.

In the short experiment, the standardized baseline reached 45.32% test accuracy, compared with 41.52% without standardization. Normalization keeps input values on a useful scale, which can make optimization more stable and help training converge. Standardization performed better here, although the short training time means that difference may not hold after longer training.

## Q2 — Basic CNN

The baseline had two convolutional layers: 32 filters of size 3 × 3 followed by 64 filters of size 3 × 3. Each convolution was followed by ReLU and 2 × 2 max pooling. The feature maps were flattened and passed through a 128-neuron ReLU layer to produce 10 class scores. The model had 545,098 parameters. Cross-entropy loss was used; it handles the output logits without needing a separate Softmax layer.

Convolutional layers learn local patterns such as edges and textures. ReLU lets the network learn non-linear patterns and is simple to compute. Pooling reduces the feature-map size. The fully connected layers combine the learned features to predict a class. More filters can help the model learn more patterns, but they also increase computation and can make overfitting more likely.

At epoch 2 of the controlled baseline run, training loss/accuracy were 1.4251/48.49%, and validation loss/accuracy were 1.5239/45.63%. Per-epoch training and validation values and plots are included in the notebook.

## Q3 — Kernel size

| Kernel | Parameters | Training time | Training accuracy | Validation accuracy | Test accuracy |
|---|---:|---:|---:|---:|---:|
| 3 × 3 | 545,098 | 22.69 s | 48.49% | 45.63% | 45.32% |
| 5 × 5 | 579,402 | 26.38 s | 48.43% | 44.63% | 44.90% |

The 5 × 5 filters have more weights and cover a wider area of the image at once. They took longer in this run and did not improve accuracy. The 3 × 3 model performed slightly better. A larger kernel does not automatically give better results.

## Q4 — Pooling

Max pooling keeps the strongest activation in each region, while average pooling keeps the average activation. Both reduce the spatial dimensions and lose some location detail.

| Pooling | Training accuracy | Validation accuracy | Test accuracy | Training time |
|---|---:|---:|---:|---:|
| Max | 48.49% | 45.63% | 45.32% | 22.69 s |
| Average | 49.64% | 46.63% | 46.17% | 15.68 s |

Average pooling did slightly better in this short run. The result is specific to these settings and does not prove that average pooling is always better for CIFAR-10.

## Q5 — Network depth

| Model | Convolutional layers | Parameters | Training time | Training accuracy | Validation accuracy | Test accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Shallow | 2 | 545,098 | 22.69 s | 48.49% | 45.63% | 45.32% |
| Deeper | 4 | 1,290,826 | 68.98 s | 46.40% | 44.13% | 43.59% |

The deeper model took about three times longer and did not perform better in this experiment. Deeper layers can combine simple patterns into more complex features, but adding layers can also make optimization harder and increase overfitting risk. Here the deeper model had lower training as well as validation accuracy, so its lower test accuracy is not clear evidence of overfitting.

## Q6 — Regularization and augmentation

| Experiment | Training accuracy | Validation accuracy | Test accuracy | Training time |
|---|---:|---:|---:|---:|
| Baseline | 48.49% | 45.63% | 45.32% | 22.69 s |
| Dropout (0.5) | 47.97% | 46.25% | 45.85% | 17.05 s |
| Weight decay (0.0001) | 53.03% | 48.25% | 48.87% | 19.10 s |
| Data augmentation | 41.64% | 40.38% | 41.42% | 23.23 s |

Weight decay gave the best validation and test accuracy in these runs. Dropout made a small improvement over the baseline on the test set. Augmentation performed worse after only two epochs; it may need more training time to help. Too much regularization can make it harder for a model to learn the training data.

The baseline had a training-to-validation accuracy gap of about 2.86 percentage points. The weight-decay model's gap was about 4.78 points. These gaps show some difference between training and validation performance, but two epochs are not enough to decide confidently whether any model is fully trained or strongly overfitting.

## Q7 — Test results

The weight-decay configuration was selected using validation accuracy, then evaluated on the official test set. In the controlled comparison, it reached 48.87% accuracy with a test loss of 1.4086. The notebook contains the confusion matrix, classification report, and examples of correct and incorrect predictions.

In the classification report, frog had the highest recall, ship had the highest F1-score, and bird and cat had the lowest F1-scores. Common confusions included cat as dog, automobile as truck, and bird as dog. Similar-looking objects and the small image size may contribute to these errors. The model still has substantial errors, and its short training means I cannot conclude it has reached its best performance.

## Q8 — Overall comparison

The table below summarizes the recorded comparison. The training/validation gap is training accuracy minus validation accuracy.

| Experiment | Parameters | Time | Train accuracy | Validation accuracy | Test accuracy | Train-validation gap |
|---|---:|---:|---:|---:|---:|---:|
| Standardized baseline, 3 × 3, max pool | 545,098 | 22.69 s | 48.49% | 45.63% | 45.32% | 2.86 pp |
| 5 × 5 kernel | 579,402 | 26.38 s | 48.43% | 44.63% | 44.90% | 3.81 pp |
| Average pooling | 545,098 | 15.68 s | 49.64% | 46.63% | 46.17% | 3.01 pp |
| Four convolutional layers | 1,290,826 | 68.98 s | 46.40% | 44.13% | 43.59% | 2.28 pp |
| Dropout (0.5) | 545,098 | 17.05 s | 47.97% | 46.25% | 45.85% | 1.72 pp |
| Weight decay (0.0001) | 545,098 | 19.10 s | 53.03% | 48.25% | 48.87% | 4.78 pp |
| No standardization | 545,098 | 17.76 s | 42.76% | 40.50% | 41.52% | 2.26 pp |
| Data augmentation | 545,098 | 23.23 s | 41.64% | 40.38% | 41.42% | 1.26 pp |

Weight decay had the highest test accuracy in this comparison. The 3 × 3 model was slightly better than the 5 × 5 model, and average pooling was slightly better than max pooling. The deeper model was slower and had lower accuracy. Weight decay also had the highest training accuracy. These conclusions are based on one short run per setting, so longer or repeated runs would be needed to be more certain.

## Phone-image evaluation

The phone-image evaluation covered 24 labeled images from eight class folders. The saved per-image results include the actual class, predicted class, confidence, and whether the prediction was correct in [`outputs/phone_image_predictions.csv`](outputs/phone_image_predictions.csv).

| Class | Correct / total |
|---|---:|
| Airplane | 2 / 3 |
| Bird | 0 / 3 |
| Cat | 0 / 3 |
| Dog | 2 / 3 |
| Frog | 0 / 3 |
| Horse | 0 / 3 |
| Ship | 3 / 3 |
| Truck | 3 / 3 |
| **Overall** | **10 / 24 (41.67%)** |

The phone-image checkpoint achieved 48.34% accuracy on the CIFAR-10 test set in its evaluation run, compared with 41.67% on these images. This evaluation used a separately saved quick weight-decay checkpoint, not the same training run as the 48.87% weight-decay result in the comparison table. The two results are reported separately because they came from different checkpoints.

The images are small in number and may differ from CIFAR-10 in background, lighting, framing, and image content, so they are not a reliable estimate of real-world performance. If the assignment requires proof that each image was personally captured with my own phone, I should only describe them as my own phone photos if that is true.

## Limitations and conclusion

The requested model choices were compared, but each comparison used only two epochs and a 7,200/800 training/validation split. The runs were not repeated with different random seeds. The results are preliminary, the model's accuracy is moderate, and the phone-image evaluation covers only 24 images. Its checkpoint is also separate from the comparison-table checkpoint.

From these measurements, weight decay was the best tested configuration. Standardization performed better than scaling pixels to [0, 1] alone. The 3 × 3 kernel and shallow network performed better than their alternatives in these runs, while average pooling had a small advantage over max pooling. Longer and repeated experiments would be needed before treating these findings as final.

## References

1. Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*. CIFAR-10 dataset: https://www.cs.toronto.edu/~kriz/cifar.html
2. LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. (1998). “Gradient-Based Learning Applied to Document Recognition.” *Proceedings of the IEEE*, 86(11), 2278–2324.
3. He, K. et al. (2016). “Deep Residual Learning for Image Recognition.” *CVPR*.
4. Paszke, A. et al. (2019). “PyTorch: An Imperative, High-Performance Deep Learning Library.” *NeurIPS*.
