# SPECDIFF-GAN: A Spectrally-Shaped Noise Diffusion GAN for Speech and Music Synthesis

### Teysir Baoueb, Haocheng Liu, Mathieu Fontaine, Jonathan Le Roux, Gaël Richard

This repository contains the official implementation [SPECDIFF-GAN: A Spectrally-Shaped Noise Diffusion GAN for Speech and Music Synthesis](https://arxiv.org/abs/2402.01753).

SpecDiff-GAN introduces an enhanced version of HiFi-GAN, a high-fidelity mel spectrogram-to-speech waveform synthesizer, by incorporating a diffusion process with spectrally-shaped noise. Audio examples are available on our [demo page](https://specdiff-gan.github.io/).

**Abstract**<br/>
Generative adversarial network (GAN) models can synthesize highquality audio signals while ensuring fast sample generation. However, they are difficult to train and are prone to several issues including mode collapse and divergence. In this paper, we introduce SpecDiff-GAN, a neural vocoder based on HiFi-GAN, which was initially devised for speech synthesis from mel spectrogram. In our model, the training stability is enhanced by means of a forward diffusion process which consists in injecting noise from a Gaussian distribution to both real and fake samples before inputting them to the discriminator. We further improve the model by exploiting a spectrally-shaped noise distribution with the aim to make the discriminator's task more challenging. We then show the merits of our proposed model for speech and music synthesis on several datasets. Our experiments confirm that our model compares favorably in audio quality and efficiency compared to several baselines.

## Training
To train the model, you can run the `train.py` script with different arguments to customize the training process. A brief description is provided for each argument within the file.

## Inference
To use inference.py, use the following command:<br/>

  ```
  python inference.py --input_wavs_dir /path/to/input_wavs --output_dir /path/to/output_directory --checkpoint_file /path/to/checkpoint_file
  ```

Optionally, if you have a file containing a list of .wav files for inference, provide its path using the `--inference_file` argument. If not specified, the script will infer from all .wav files in the directory specified by `--input_wavs_dir`.

## References
- [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://github.com/jik876/hifi-gan/)
- [UnivNet: A Neural Vocoder with Multi-Resolution Spectrogram Discriminators for High-Fidelity Waveform Generation](https://arxiv.org/abs/2106.07889)
- [Diffusion-GAN: Training GANs with Diffusion](https://github.com/Zhendong-Wang/Diffusion-GAN)
- [SpecGrad: Diffusion Probabilistic Model based Neural Vocoder with Adaptive Noise Spectral Shaping](https://arxiv.org/abs/2203.16749)
