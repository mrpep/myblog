---
title: "Exploiting the stereo field to separate music sources"
description: Separating sources by undoing panning.
tags: ["dsp", "audio", "spectrograms", "stereo", "source separation", "python"]
date: 2023-04-15T00:33:05-03:00
draft: false
author: Leonardo Pepino
authorLink: /about
math: true
---

### Stereo and panning

Stereophonic sound (stereo) is a method of sound reproduction where 2 independent channels
are used to recreate the multi-directional aspect of sound. If we want a sound to feel placed on the left of the listener,
then during the mixing process, we can increase the level of the sound for the left channel and decrease it for the right one.
This process of artificially placing sources in the stereo field by assigning different levels to each channel is called **panning**.
In this article I am going to show one way to undo the panning from a stereo recording.
This way, it is possible to separate sources that are placed in a particular stereo direction where there are no other sources.
In modern stereo mixing it is not common to place only one source in a particular direction, but multiple sources share some range of the stereo field.
However, in early rock stereo mixes of, for example, The Beatles or Jimi Hendrix,
because of limitations of mixing devices, the mono tracks of different instruments were hard panned to the left, right or center and we'll see that in this cases,
an almost perfect separation of the original tracks is possible using simple digital signal processing techniques.

So, let's imagine we have a source {{< raw >}}\(S\){{< /raw >}}, and two channels {{< raw >}}\(L, R\){{< /raw >}}. Then:

{{< raw >}}
\[ L=\alpha*S \\ R=(1-\alpha)*S \]
{{< /raw >}}

where {{< raw >}}\(\alpha\){{< /raw >}} is a real number between 0 and 1. We are modelling the panning as linear, although we know it is determined by the electronics of the pan potentiometer used in the mixing device.
In this case if {{< raw >}}\(\alpha=0\){{< /raw >}}, then the source will be hard panned to the right. In the other hand, if
{{< raw >}}\(\alpha=1\){{< /raw >}}, then the source will be hard panned to the left. If {{< raw >}}\(\alpha=0.5\){{< /raw >}}, then
the source will be panned to the center and will be equally loud for the left and right channels.

### Mid-side processing

Let's discuss a first approach commonly used to remove vocals from a stereo song. Let's suppose the vocals are panned to the center (which is a common practice), so
{{< raw >}}\(\alpha=0.5\){{< /raw >}} and if we do L - R we get:

{{< raw >}}
\[ L-R = 0.5*S - 0.5*S = 0\]
{{< /raw >}}

So every source panned to the center will get removed. Other sources that are not center panned will remain. This new track is usually called Side.
Then if instead of doing L-R, we do L+R, we get what is usually called the Mid.

Let's try this technique with a song:

```python
from IPython.display import Audio
import librosa

!wget https://archive.org/download/therollingstonessympathyforthedevilofficiallyricvideo_20190702/The%20Rolling%20Stones%20-%20Sympathy%20For%20The%20Devil%20%28Official%20Lyric%20Video%29.ogg
x,fs = librosa.core.load('The Rolling Stones - Sympathy For The Devil (Official Lyric Video).ogg',mono=False,sr=None)
side = x[0] - x[1]
Audio(side,rate=fs)
```

Here it's a 30s excerpt of the resulting track:
{{< music url="audio/sympathy-side.wav" name="Sympathy for the devil" artist="The Rolling Stones">}}

Conversely, we can listen to the mid track:

```python
mid = x[0] + x[1]
Audio(mid,rate=fs)
```
{{< music url="audio/sympathy-mid.wav" name="Sympathy for the devil" artist="The Rolling Stones">}}

Not bad! However we can notice that not only the vocals got removed but also the bass, as both instruments were panned to the center.
Now, what if we want to separate a source that is only in the left channel? Well, an easy answer would be, just listen to the left channel.
However, there might be other sources that are in the center or other directions, that will get its way into the left channel too.
This means that although a source is hard panned left or right, most of times it won't be enough to just use the corresponding channel as other sources will interfere.

Let's try to separate the vocals in With a Little Help from my friends (The Beatles).

We can try the approaches discussed above and the results are these:

{{< music url="audio/somebody-mid.wav" name="With a little help from my friends - Mid" artist="The Beatles">}}
{{< music url="audio/somebody-side.wav" name="With a little help from my friends - Side" artist="The Beatles">}}

So it seems the mid-side decomposition is not useful in this case as the vocals are not panned to the center.
Let's try something a bit more sophisticated.

### "Unpanning" in the time-frequency domain

Suppose now that we have the L and R tracks, and we want to find the alpha of a source from them.
If we divide L by R we get:

{{< raw >}}
\[ \frac{L}{R} = \frac{\alpha}{1-\alpha}\]
{{< /raw >}}

We will define {{< raw >}}\(R = \frac{L}{R}\){{< /raw >}} and with a bit of algebraic manipulation we can find that:

{{< raw >}}
\[ \alpha = \frac{R}{1+R}\]
{{< /raw >}}

Now that we have {{< raw >}}\(\alpha\){{< /raw >}}, we can use it to mask out everything that is not in the desired range of {{< raw >}}\(\alpha\){{< /raw >}}.
For example, if we want to keep only the sources panned to the center, we would keep only the parts of audio with {{< raw >}}\(\alpha\){{< /raw >}} between, for example, 0.45 and 0.55.

But there is a problem, if we work in the time domain we can only mask complete regions of time. Also, if there is a bit of delay in one of the channels, then the formulation doesn't hold true.
So, a better approach is to work in the time-frequency domain instead. Let's do it!

The first step is to calculate spectrograms of the L and R channels of our song:

```python
import librosa

!wget https://archive.org/download/cd_sgt.-peppers-lonely-hearts-club-band_the-beatles/disc1/02.%20The%20Beatles%20-%20With%20a%20Little%20Help%20From%20My%20Friends_sample.mp3
x,fs = librosa.core.load('02. The Beatles - With a Little Help From My Friends_sample.mp3',mono=False,sr=None)

XL = librosa.core.stft(x[0])
XR = librosa.core.stft(x[1])

```

We can plot the magnitude of the spectrograms:

```python
hop_size=512
n_freq=1025
xlabels = [np.round(x,1) for x in np.linspace(0,XL.shape[1],5)*hop_size/fs]
xpos = np.linspace(0,XL.shape[1],5)
ylabels = [int(np.round(x,0)) for x in np.linspace(0,XL.shape[0],5)*fs/(2*n_freq)]
ypos = np.linspace(0,XL.shape[0],5)
fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(20,5))
ax[0].imshow(np.log(np.abs(XL)+1e-12), aspect='auto', origin='lower')
ax[1].imshow(np.log(np.abs(XR)+1e-12), aspect='auto', origin='lower')
_=ax[0].set_xticks(xpos)
_=ax[0].set_xticklabels(xlabels)
_=ax[0].set_yticks(ypos)
_=ax[0].set_yticklabels(ylabels)
ax[0].set_xlabel('Time (s)')
ax[0].set_ylabel('Frequency (Hz)')
ax[0].set_title('Left channel')
_=ax[1].set_xticks(xpos)
_=ax[1].set_xticklabels(xlabels)
_=ax[1].set_yticks(ypos)
_=ax[1].set_yticklabels(ylabels)
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('Frequency (Hz)')
ax[1].set_title('Right channel')
```

{{< figure src="images/LR_stft.png" title="Left and right channels magnitude spectrograms" >}}

Now, we can calculate R and from that matrix get {{< raw >}}\(\alpha\){{< /raw >}} and plot it.

```python
ratio = np.abs(XL)/(1e-9+np.abs(XR))
alpha = ratio/(1+ratio)

plt.figure(figsize=(20,5))
plt.imshow(alpha,origin='lower',cmap='seismic',aspect='auto')
plt.colorbar()
plt.xticks(ticks=xpos, labels=xlabels)
plt.yticks(ticks=ypos, labels=ylabels)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Alpha values')
```

{{< figure src="images/alphas.png" title="Values of alpha for the spectrograms" >}}

We can notice that the red color (which would correspond to left panning) seems to show one isolated source, and the blue color another one.
Also, the whites (center panning), seem to show other sources.

Let's try to separate the reds and listen to them! For that we are going to do 3 things:
* Create a mask to keep the time-frequency bins with {{< raw >}}\(\alpha\){{< /raw >}} between 0.55 and 1. We'll also keep only those time-frequency bins where magnitude is higher than a threshold.
* Apply the mask by multiplying it with the left and right channel complex spectrograms.
* Go back to time domain by applying the inverse Short Term Fourier Transform (ISTFT)
The code to achieve this is:

```python
#Here we create the mask
mask = np.logical_and(np.logical_and(alpha>0.55,alpha<1),(np.abs(XL)+np.abs(XR))>0.01)

#Here we apply the mask
XL_masked = XL*mask
XR_masked = XR*mask

#Here we go back to time domain and listen to the resulting audio
xl_filtered = librosa.core.istft(XL_masked)
xr_filtered = librosa.core.istft(XR_masked)
output = np.array([xl_filtered,xr_filtered])
Audio(output, rate=fs)
```

And this is the result
{{< music url="audio/somebody-alpha.wav" name="With a little help from my friends" artist="The Beatles">}}

Now you have a tool to unveil sounds hidden in the stereo field of old rock recordings. Give it a try with some good old Jimi Hendrix, The Doors, The Beatles, etc...

