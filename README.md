# GANymede

Trying some implementations of GANs for audio in Keras.

## WaveGAN

WaveGAN implements a structure that enables working on raw audio.

- Paper : [Synthesizing Audio with Generative Adversarial Networks](https://arxiv.org/abs/1802.04208)

### Advancement
- [x] Generator structure
- [x] Discriminator structure
- [x] Loss function : [WGAN-GP](https://arxiv.org/abs/1704.00028)
- [ ] Train function
- [x] Datasets : 
- - [ ] [Speech Commands](https://arxiv.org/abs/1804.03209)
- - [x] [Speech Commands Zero through Nine (SC09)](http://deepyeti.ucsd.edu/cdonahue/wavegan/data/sc09.tar.gz). Source : [here](https://github.com/chrisdonahue/wavegan)
- - [x] [Drum sound effects](http://deepyeti.ucsd.edu/cdonahue/wavegan/data/drums.tar.gz). Source : [here](https://github.com/chrisdonahue/wavegan)
- - [x] [Bach piano performances](http://deepyeti.ucsd.edu/cdonahue/wavegan/data/mancini_piano.tar.gz). Source : [here](https://github.com/chrisdonahue/wavegan)

## GANSynth

- Paper : [GANSynth: Adversarial Neural Audio Synthesis](https://arxiv.org/abs/1902.08710)

> Incoming implementation
