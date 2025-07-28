# Welcome to TokYy FrameWork [!]

> !!! TokYy is a work-in-progress framework !!!

TokYy is a framework developed upon the PyTorch framework to build different Deep Learning models, focusing on computer vision models.


# File structures

| dir_name | file_name    | class                  | class_description                                                                                         |
|----------|--------------|------------------------|-----------------------------------------------------------------------------------------------------------|
| datasets | nyudepthv2   | Dataset                | Parses locally downloaded NYU-Depth-V2 dataset. Used to create dataset loaders.                           |
| losses   | losses       | AWLoss                 | [ `0.1 * l1_loss + gradient_loss + ssim_loss` ] Used for loss between same sized images.                    |
| .        | checkpointer | Checkpointer           | Monitors, saves and loads model information.                                                              |
| .        | metrics      | Metrics                | Enum holding the metrics allowed to be computed.                                                          |
|          |              | Metric                 | Computes metrics.                                                                                         |
| .        | trainer      | Trainer                | Contains all steps to train a model, from loading dataset, to training on it.                             |
| .        | utils        | LogType                | Used by log_message function to log messages inside the console.                                          |
| models   | resunet      | DEPRECIATED            | DEPRECIATED                                                                                               |
|          | models       | ResBlock               | Residual CNN Block                                                                                        |
|          |              | SEBlock                | Squeeze & Excitation Network                                                                              |
|          |              | ChannelAttentionModule | Used in CBAM                                                                                              |
|          |              | SpatialAttentionModule | Also used in CBAM, after ChannelAttentionModule.                                                          |
|          |              | CBAMBlock              | Convolutional Block with Attention Module (channel + spatial)                                             |
|          |              | ResSEBlock             | Residual block implementing SEBlock                                                                       |
|          |              | ResCBAMBlock           | Residual block implementing CBAMBlock                                                                     |
|          |              | AttentionGate          | Attention gate                                                                                            |
|          |              | Encoder                | Encoder part of an U-Net using ResCBAMBlock by default.                                                   |
|          |              | DecoderBlock           | Block in the decoder part of the U-Net (Does not implement full decoder) using Attention Gates by default |
|          |              | ResUNet                | Residual U-Net with Attention Gates using Encoder and DecoderBlock.                                       |
