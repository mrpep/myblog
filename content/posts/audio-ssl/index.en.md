---
title: "Audio representation learning"
tags: ["self-supervised","audio"]
date: 2023-04-15T01:13:43-03:00
draft: true
math: true
raw: true
---

In self-supervised learning, the idea is to define a pretext task
based on unlabeled inputs to produce representations.
These representations can potentially be used across a wide range of tasks,
surpassing in many cases the performance of fully supervised models.
One of the main motivations behind self-supervised learning is that no labels are required,
which unlocks the use of datasets which are orders of magnitude larger than the ones available in supervised settings.

#### Information restoration approaches

##### Audio word2Vec

Audio word2vec represents a variable length audio segment with a single fixed size vector by using
a sequence to sequence autoencoder. The authors explore 2 pretext tasks: autoencoding and denoising autoencoding.
They trained on Librispeech using 13 MFCCs as input and target sequences. The resulting embeddings
were evaluated in a query by example task, showing better performance than using DTW. 
Some interesting results, similar to Word2Vec in NLP are achieved, but measuring phoneme similarity instead of word meaning.
Moreover, in spite of using LSTMs which are known to "forget" the first elements in the sequence, changes in
the beggining or end of the audio segment are noticed in the embedding space, as seen in the figures.

{{< figure src="images/audio2vec_1.png" title="Change of the first phoneme" >}}
{{< figure src="images/audio2vec_2.png" title="Change of the last phoneme" >}}

#### Multi-view invariance approaches

##### Audio-Audio invariance

Audio representations can be learned by defining invariances. We can define a transformation {{< raw >}}\(T(x)\){{< /raw >}} for 
an audio {{< raw >}}\(x\){{< /raw >}}, and a representation extractor {{< raw >}}\(F(x)\){{< /raw >}}, and we would like that under 
some transformations {{< raw >}}\(F(T(x)) = F(x)\){{< /raw >}}. For example, {{< raw >}}\(T(x)\){{< /raw >}} could be a pitch-shifter. 
In that case, we would want our representation to be invariant to pitch, which might be useful for speech recognition, 
but harmful for music transcription or emotion recognition from speech.

Several works have used triplet loss to learn audio invariances. For example, TRILL [7] samples triplets {{< raw >}}\((x_i,x_j,x_k)\){{< /raw >}},
such that {{< raw >}}\(||j-i||\leq\tau\){{< /raw >}} and {{< raw >}}\(||k-i||\gt\tau\){{< /raw >}}, where {{< raw >}}\(x\){{< /raw >}} are fixed size windows taken from an audio.
This way, segments of audio that are closer in time, are also mapped closer in the representation. The motivation behind this is that some
aspects like speaker identity are stable in time.

In [8], 

##### Audio-text invariance

Several works have explored using triplet loss to learn acoustic representations.
In [2], the triplet consists of an acoustic embedding, the representation of the word, and the representation of a wrong word.
The acoustic embedding is obtained from the output of a CNN pretrained to recognize words, while the word representations are
trained from scratch using letter n-grams as inputs to a 3 layer MLP. Once the model is trained, the acoustic embedding of a 
word will be close to the representation of that word, which is useful to improve ASR models.
{{< figure src="images/bengio2014_1.png" title="Architecture proposed in [2]" >}}

##### Audio-visual invariance

Other works have explored using audio-visual correspondences to learn acoustic embeddings. 

For example, SoundNet [6] learns acoustic representations by using pretrained image classifiers as teachers. A deep CNN taking as
input the raw waveforms has to predict the representations of the corresponding image frame. The model is trained using the KL divergence
between the outputs of the acoustic model and the image models. 2 image models are used as teachers: Imagenet CNN and Places CNN.
Acoustic representations are extracted from an intermediate layer and used for acoustic event detection by training a linear SVM.

Similarly, in [3] a visual and an audio deep neural networks are trained jointly for the task of detecting if the audio and image correspond to the same video. For example,
if the image shows a violin player, then the audio subnetwork will have to learn how a violin sounds to tell if
there is correspondence. This way, the audio subnetwork learns in a self-supervised way acoustic concepts.
The audio subnetwork consists of a VGG-style CNN that takes 1 second log-spectrogram and outputs a 512-D vector.
The visual subnetwork follows the same pattern but uses a video frame as input. The 2 vectors are concatenated and further processed 
by a fusion network consisting of 2 fully-connected layers. The learned acoustic representations set the state of the art for event detection in ESC-50 at that time.
{{< figure src="images/l3_1.png" title="Model proposed in [3]" >}}
{{< figure src="images/l3_2.png" title="Architectures used in [3]" >}}

In a follow-up work [4], the authors explored the effect of some hyperparameters in their setup. The main findings are:
* Using mel-spectrograms instead of linear spectrograms as input improves performance. Particularly in a acoustic scene classification task,
using 256 channels instead of 128 gives a significant improvement.
* In spite of pretraining in music-only videos, the learned representations are useful for event recognition downstream tasks. 
The mismatch between pretraining and finetuning data doesn't seem to hurt performance compared to using environmental videos as pretraining data.
* L3-Net outperforms VGGish [5] and SoundNet [6] in environmental sound classification in spite of using 10x less parameters and 100x less data than VGGish.
* Using larger training datasets improve downstream performance, although from 40M samples it seems that performance gets saturated as shown below.

{{< figure src="images/l3_3.png" title="Downstream accuracy vs number of pretraining samples [4]." >}}

#### References
[1] Chung, Y. A., Wu, C. C., Shen, C. H., Lee, H. Y., & Lee, L. S. (2016). Audio word2vec: Unsupervised learning of audio segment representations using sequence-to-sequence autoencoder. arXiv preprint arXiv:1603.00982.

[2] Bengio, S., & Heigold, G. (2014). Word embeddings for speech recognition.

[3] Arandjelovic, R., & Zisserman, A. (2017). Look, listen and learn. In Proceedings of the IEEE international conference on computer vision (pp. 609-617).

[4] Cramer, J., Wu, H. H., Salamon, J., & Bello, J. P. (2019, May). Look, listen, and learn more: Design choices for deep audio embeddings. In ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 3852-3856). IEEE.

[5] Hershey, S., Chaudhuri, S., Ellis, D. P., Gemmeke, J. F., Jansen, A., Moore, R. C., ... & Wilson, K. (2017, March). CNN architectures for large-scale audio classification. In 2017 ieee international conference on acoustics, speech and signal processing (icassp) (pp. 131-135). IEEE.

[6] Aytar, Y., Vondrick, C., & Torralba, A. (2016). Soundnet: Learning sound representations from unlabeled video. Advances in neural information processing systems, 29.

[7] Shor, J., Jansen, A., Maor, R., Lang, O., Tuval, O., Quitry, F. D. C., ... & Haviv, Y. (2020). Towards learning a universal non-semantic representation of speech. arXiv preprint arXiv:2002.12764.

[8] Jansen, A., Plakal, M., Pandya, R., Ellis, D. P., Hershey, S., Liu, J., ... & Saurous, R. A. (2018, April). Unsupervised learning of semantic audio representations. In 2018 IEEE international conference on acoustics, speech and signal processing (ICASSP) (pp. 126-130). IEEE.