---
title: "Audio generation using Language Models"
tags: ["audio","synthesis","encodec","encodecmae","gpt2"]
date: 2023-10-23
draft: false
math: true
raw: true
---

## ¿Qué es un modelo de lenguaje?

Los modelos de lenguaje, como ChatGPT, Llama y Mistral, son un tema candente en la actualidad y parece que todo el mundo está usándolos; pero, ¿podemos generar audio utilizando modelos de lenguaje? En este artículo, te mostraré algunos de los experimentos que realicé para generar audio utilizando GPT2, EnCodec y EnCodecMAE. Pero primero, hablemos de modelos de lenguaje (LMs).

Los modelos de lenguaje estiman probabilidades de secuencias de palabras dadas un corpus. ¿Cuál es la probabilidad de una secuencia? Dada una secuencia {{< raw >}}\(S\){{< /raw >}} de longitud {{< raw >}}\(T\){{< /raw >}}: {{< raw >}}\( [x_1,x_2,...,x_T]\){{< /raw >}}, su probabilidad se calcula como la probabilidad conjunta de cada elemento en la secuencia:

{{< raw >}}
\[ P(S) = P(x_1,x_2,...,x_T)\]
{{< /raw >}}

Usando la regla de la cadena de probabilidad, y asumiendo un ordenamiento en los elementos de la secuencia, la probabilidad de la secuencia puede ser expresada como:
{{< raw >}}
\[ P(S) = P(x_1)P(x_2|x_1)P(x_3|x_1,x_2),...,P(x_T|x_1,...,x_{T-1})\]
{{< /raw >}}
Expliquemos esto con un ejemplo concreto. Digamos que tenemos la siguiente oración:

<p style="text-align: center;"><b>"Juego al fútbol"</b></p>

El primer paso es convertir esta oración en una secuencia. Hay muchas maneras posibles de hacerlo:

<p style="text-align: center;"><b>"[Juego, al, fútbol]"</b></p>

o

<p style="text-align: center;"><b>"[Juego, al, fut#, #bol]"</b></p>

o

<p style="text-align: center;"><b>"[J,u,e,g,o, ,a,l, ,f,u,t,b,o,l]"</b></p>

Tokenizar es el proceso de transformar un corpus en una secuencia de símbolos. Estos símbolos pueden ser palabras, como en el primer ejemplo; o permitir sub-palabras como descomponer fútbol en fut# y #bol; o podrían ser caracteres como en el último ejemplo. Utilicemos la tokenización por palabras y calculemos la probabilidad de la secuencia. De acuerdo a la regla de la cadena tenemos:

{{< raw >}}
\[ P(S) = P(x_1)P(x_2|x_1)P(x_3|x_1,x_2),...,P(x_T|x_1,...,x_{T-1})\]
\[ P(S) = P(Juego)P(al|Juego)P(futbol|Juego,al)\]
{{< /raw >}}

Cada término de la ecuación nos dice la probabilidad de una palabra dadas las anteriores. Si entrenamos una red neuronal para predecir la siguiente palabra dadas las anteriores, y usamos entropía cruzada como función de costo, luego las salidas de la red corresponderan con cada término de la ecuación. El detalle importante es que nuestros modelos deben estar restringidos a solo mirar el pasado; si tiene acceso a tokens del futuro, las probabilidades no estarán más condicionadas solamente en las palabras anteriores rompiéndose la regla de la cadena.
Para redes recurrentes (RNNs) esta restricción es inherente al modelo ya que las palabras se procesan en orden secuencial. En el caso de transformers, hay que aplicar una máscara de atención para que esta se calcule solamente entre cada query y los key-values pasados.
Los métodos más tradicionales, como n-gramas, simplifican los términos de la ecuación y condicionan solo con un número fijo de valores pasados (n-1 en n-gramas). Por otro lado, redes convolucionales causales como Wavenet (Oord et al., 2016) pueden ser utilizadas como modelos de lenguaje, siendo el campo receptivo el equivalente al n en n-gramas.

## Tokenización de audio

Esta idea se puede extender para modelar cualquier secuencia, no solo texto. Tratemos de modelar audio!
El primer paso es convertir audio en una secuencia {{< raw >}}\(S\){{< /raw >}}. El audio en si mismo es una secuencia
de valores discretos, donde la frecuencia de muestreo define cuántos valores hay en un segundo. Para habla, una frecuencia de muestreo común es 16000 Hz, y para música 44100 Hz. Esto significa que si tenemos un segundo de audio y queremos predecir la siguiente muestra, el modelo debe tener en cuenta los 44100 valores previos. Esto es muy costoso computacionalmente para transformers y RNNs. Una manera de resolver esto es utilizar una representación más compacta de audio. Vamos a utilizar [EnCodec](https://github.com/facebookresearch/encodec) (Défossez, Copet, Synnaeve & Adi, 2022), que es un codec neuronal para audio (algo asi como un MP3), que mediante una red neuronal profunda es capaz de comprimir un segundo de audio en una secuencia de sólo 75 elementos.

### EnCodec
{{< figure src="images/encodec.png" title="EnCodec architecture. From (Défossez, Copet, Synnaeve & Adi, 2022)" >}}

Como puede verse en la figura, EnCodec tiene un encoder y un decoder, y su objetivo es reconstruir el audio de entrada (es un autoencoder). La función de costo consiste de una suma pesada de varios términos: un costo de reconstrucción en el dominio temporal {{< raw >}}\(l_t\){{< /raw >}}; a nivel espectrograma {{< raw >}}\(l_s\){{< /raw >}} y una función de costo adversaria para reducir artefactos. El detalle importante es que este autoencoder tiene un cuello de botella muy reducido: el encoder reduce el audio de entrada de 24000 Hz a solo 75 Hz (320x). Esto parece un montón de compresión, sin embargo hay que tener en cuenta que esos 75 elementos por segundo tienen 128 dimensiones. Por lo tanto, en términos de información, la reducción es solo de 2.5X.

Para comprimir aún más, EnCodec cuantiza el cuello de botella. La idea es que una capa de cuantización mapea cada uno de los vectores 128-D a uno de 1024 valores posibles. Esto se hace aprendiendo un codebook, el cual es una tabla con 1024 vectores 128-D (códigos). Luego, durante inferencia, la salida del encoder se compara con cada uno de los códigos y se transforma en el índice correspondiente al más cercano. 1024 valores posibles se pueden representar con 10 bits. Esto significa que un segundo de audio puede representarse con 75 valores de 10 bits, lo que nos da un bitrate de 750 bps.
Nuestro audio sin comprimir tenía una frecuencia de muestreo de 24000 Hz con profundidad de 16 bits, lo que nos da un bitrate de 384 Kbps. Estamos comprimiendo por 512! Sin embargo, hay un problema: la calidad del audio comprimido va a ser muy mala.

Para resolver este problema, EnCodec usa residual vector quantization (RVQ), que significa que la diferencia entre la salida del encoder y el código más cercano del primer cuantizador va a ser cuantizada por un segundo cuantizador, y asi sucesivamente. Esto permite utilizar múltiples cuantizadores que refinen la salida de cuantizadores previos. Si usamos 8 cuantizadores, podemos obtener una calidad de audio decente y estaríamos representando 1 segundo de audio con solo 8 secuencias de 75 valores de 10 bits. Esto da un bitrate de 6 Kbps y una compresión de 64X.

Es muy fácil comprimir (o tokenizer) y descomprimir con EnCodec en Python:

```python
from encodec import EncodecModel
import librosa
import torch

def tokenize(filename, model, sr=24000, num_quantizers=8):
    with torch.no_grad():
        x, fs = librosa.core.load(filename, sr=sr)
        codes = model.encode(torch.from_numpy(x)[None,None,:])[0][0]
    return codes

def detokenize(codes, model):
    with torch.no_grad():
        decoded_x = model.decode([(codes,None)])
    return decoded_x.detach().cpu().numpy()[0,0]

NUM_QUANTIZERS = 8

model = EncodecModel.encodec_model_24khz()
codes = tokenize('/mnt/hdd6T/jamendo/00/565800.mp3', model)
reconstruction_8q = detokenize(codes[:,:NUM_QUANTIZERS], model)
```

Estos son algunos ejemplos de audios comprimidos utilizando distinto número de cuantizadores:
{{< music url="audio/original.wav" name="Uncompressed, bitrate=384 Kbps" artist="Unknown">}}
{{< music url="audio/1q.wav" name="Q=1, bitrate=750 bps" artist="Unknown">}}
{{< music url="audio/2q.wav" name="Q=2, bitrate=1.5 Kbps" artist="Unknown">}}
{{< music url="audio/4q.wav" name="Q=4, bitrate=3 Kbps" artist="Unknown">}}
{{< music url="audio/8q.wav" name="Q=8, bitrate=6 Kbps" artist="Unknown">}}
{{< music url="audio/16q.wav" name="Q=16, bitrate=12 Kbps" artist="Unknown">}}
{{< music url="audio/32q.wav" name="Q=32, bitrate=24 Kbps" artist="Unknown">}}

### Patrones multi-secuencia

{{< figure src="images/codebook_patterns.png" title="Codebook patterns. From (Copet et al., 2023)" >}}

Pareciera que ya tenemos una manera de tokenizar audios de forma eficiente, simplemente los comprimimos utilizando EnCodec y eso nos da una secuencia de 75*cantidad de cuantizadores tokens por segundo.
Sin embargo todavía tenemos un problema, relacionado con el RVQ, ya que en vez de tener una sola secuencia de tokens, ahora tenemos tantas como cuantizadores utilicemos. En los experimentos utilice 8 cuantizadores, por lo tanto tenemos 8 secuencias. Si las secuencias fueran independientes, podríamos predecirlas en paralelo. De la misma forma que predecimos la siguiente palabra dadas las anteriores, podemos predecir el siguiente valor para los 8 cuantizadores. El problema es que, por la forma en la que se define RVQ, la salida del segundo cuantizador depende de la salida del primer cuantizador, ya que está modelando el error cometido por este. Por lo tanto, asumir independencia entre cuantizadores es subóptimo. Hay distintas formas de organizar las secuencias para resolver este problema, y las mismas se discuten en (Copet et al., 2023):

- **Flattening pattern**: la idea es intercalar las secuencias de largo {{< raw >}}\(T\){{< /raw >}} de cada uno de los {{< raw >}}\(Q\){{< /raw >}} cuantizadores. {{< raw >}}\(x_{ij}\){{< /raw >}} es el elemento correspondiente al tiempo {{< raw >}}\(i\){{< /raw >}} y cuantizador {{< raw >}}\(j\){{< /raw >}}:

{{< raw >}}
\(S=[x_{11},x_{12},x_{1Q},x_{21},x_{22},...,x_{2Q},...,x_{T1},x_{T2},...,x_{TQ}]\)
{{< /raw >}}

Este patrón permite al modelo predecir el siguiente elemento basado en las salidas de los cuantizadores previos para el instante de tiempo actual, y todas las salidas de instantes previos de tiempo. De esta forma, la dependencia entre cuantizadores puede ser modelada. El problema es que ahora la secuencia es {{< raw >}}\(Q\){{< /raw >}} veces más larga, haciéndolo un patrón poco eficiente computacionalmente.

- **Parallel pattern**: este patrón es opuesto al de flattening; los cuantizadores se asumen independientes y todas las secuencias se predicen en paralelo. Este patrón no incrementa el largo de la secuencia pero es subóptimo ya que asume independencia entre cuantizadores.

- **Vall-E pattern**: este patrón se utiliza en Vall-E (Wang et al., 2023). El modelo devuelve la secuencia correspondiente al primer cuantizador, y luego predice el resto de los cuantizadores en paralelo. Esto significa que los cuantizadores se asumen dependientes del primer cuantizador pero independientes entre si. En este caso la secuencia tiene largo {{< raw >}}\(2T\){{< /raw >}}, y la dependencia entre el primer cuantizador y el resto es modelada. Esto tiene algo de sentido ya que el primer cuantizador reconstruye la mayor parte de la señal, mientras que el resto agrega detalle.

- **Delay pattern**: este es el patrón utilizado en MusicGen (Copet et al., 2023). Como en el parallel y Vall-E pattern, se predicen múltiples secuencias en paralelo. Sin embargo, tanto la dependencia en el tiempo como entre cuantizadores es modelada, mientras que el largo de la secuencia no se incrementa significativamente. Esto se logra haciendo un delay para cada secuencia. Por ejemplo, si miramos a la Figura tenemos:
    - En el primer paso {{< raw >}}\(s_1\){{< /raw >}}, solo se genera la salida del primer tokenizador {{< raw >}}\(k_1\){{< /raw >}}.
    - En el segundo paso {{< raw >}}\(s_2\){{< /raw >}}, tenemos acceso a la salida del primer tokenizador en {{< raw >}}\(t_1\){{< /raw >}}, por lo tanto podemos producir 2 salidas más: {{< raw >}}\(t_2\){{< /raw >}} para el cuantizador {{< raw >}}\(k_1\){{< /raw >}}, y {{< raw >}}\(t_1\){{< /raw >}} para el segundo cuantizador {{< raw >}}\(k_2\){{< /raw >}}.
    - En el tercer paso {{< raw >}}\(s_3\){{< /raw >}}, tenemos acceso a las 2 primeras salidas del primer tokenizador y a la primera del segundo. Podemos entonces generar la tercer salida del primer tokenizador, la segunda del segundo, y la primera del tercero.
    - And so on...

El siguiente fragmento de código aplica el delay pattern y también lo elimina:

```python
def roll_along(arr, shifts, dim):
    #From: https://stackoverflow.com/a/76920720
    assert arr.ndim - 1 == shifts.ndim
    dim %= arr.ndim
    shape = (1,) * dim + (-1,) + (1,) * (arr.ndim - dim - 1)
    dim_indices = torch.arange(arr.shape[dim], device=arr.device).reshape(shape)
    indices = (dim_indices - shifts.unsqueeze(dim)) % arr.shape[dim]
    
    return torch.gather(arr, dim, indices)

def apply_delay_pattern(codes):
    codes = torch.nn.functional.pad(codes+1,(0,codes.shape[1]-1))
    codes = roll_along(codes, torch.arange(0,codes.shape[1], device=codes.device)[None,:].tile((codes.shape[0],1)), 2)
    return codes

def unapply_delay_pattern(codes):
    codes = roll_along(codes, -torch.arange(0,codes.shape[1], device=codes.device)[None,:].tile((codes.shape[0],1)), 2)
    codes = codes[:,:,:-codes.shape[1]]
    return codes
```
{{< figure src="images/delay2.png" title="Result of applying the delay pattern" >}}

Puede verse que este approach solo agrega {{< raw >}}\(Q-1\){{< /raw >}} elementos a la secuencia original (en este caso 7).

## Turning GPT2 into an audio generator

GPT2 (Radford et al., 2019) is the predecessor of GPT3, and it's actually a lot smaller so it can fit in a consumer-grade GPU. It's basically a transformer with the causal attention mask that allows us to use it for language modelling.
Using it in Python is very easy, thanks to HuggingFace transformers:
```python
from transformers import GPT2Model

gpt = GPT2Model.from_pretrained('gpt2')
```

### Adapting GPT2 to our inputs
We will have to make some modifications to this model to make it work with our inputs:
- The model has to predict as many outputs per timestep as quantizers. Each quantizer can have 1024 different values, and also can be empty because of the delay pattern.
This means 1025 possible classes for each quantizer. We can make a linear output classifier with 1025*8 neurons and then reshape it as a tensor of probabilities with shape {{< raw >}}\((batch\_size, 8, 1025)\){{< /raw >}}.
- The inputs to GPT2 can be discrete (corresponding to the units defined by its pretrained tokenizer), or continuous.
When the input are discrete tokens, a lookup table is learned by the model to turn each token into a continuous vector.
In our case it's a bit more complicated as we have 8 discrete tokens per timestep. What we do is to learn a lookup table with 8*1025 possible values and map each quantizer value.
Then, the 8 retrieved vectors are summed and used as the input to GPT2.

### Prompting with EnCodecMAE
Also, we want to have some control over the generated audio at inference time. One way to achieve this is by prepending a prompt to the input tokens.
The prompt could be a sequence of words describing the audio, an image, another audio, etc...
I've been working in ways to represent audios recently, and proposed [EnCodecMAE](https://github.com/habla-liaa/encodecmae) (Pepino, Riera & Ferrer, 2023), so I'll use these audio vectors as prompt.
EnCodecMAE is a general audio representation model, which is trained in a similar way to BERT. It was trained with a mixture of speech, music and general audio datasets. The discrete targets to reconstruct
from the masked inputs were the EnCodec tokens of the unmasked audio.

{{< figure src="images/encodecmae2.png" title="EnCodecMAE architecture. From (Pepino, Riera & Ferrer, 2023)" >}}

### Putting everything together

Now, we can make a Pytorch Lightning module to encapsulate everything in a class:
```python
import pytorch_lightning as pl

class EnCodecGPT(pl.LightningModule):
    def __init__(self, num_q=8, optimizer=None, lr_scheduler=None):
        super().__init__()
        self.encodec_model = EncodecModel.encodec_model_24khz()
        self.num_q = num_q
        self.num_codes = 1024
        self.vocab_size = self.num_q * (self.num_codes + 1)
        self.gpt = GPT2Model.from_pretrained('gpt2')
        self.vocab_embed = torch.nn.Embedding(self.vocab_size, self.gpt.embed_dim)
        self.classification_head = torch.nn.Linear(self.gpt.embed_dim, self.vocab_size)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def forward(self, x, prompt=None):
        #Encodec tokens:
        with torch.no_grad():
            codes = self.encodec_model.encode(x)[0][0][:,:self.num_q]
        #Delay pattern:
        codes = apply_delay_pattern(codes)
        #Offset each quantizer by 1025 and pass through LUT:
        input_vocab_embed = torch.arange(self.num_q, device=x.device)[None,:,None]*(self.num_codes + 1) + codes
        gpt_in = self.vocab_embed(input_vocab_embed).sum(axis=1)
        #Prepend prompt:
        if prompt is not None:
            gpt_in = torch.cat([prompt[:,None,:],gpt_in],axis=1)
        #Pass through GPT:
        gpt_out = self.gpt(inputs_embeds = gpt_in)
        #Make classification:
        preds = self.classification_head(gpt_out['last_hidden_state'])
        preds = preds.view(preds.shape[0],preds.shape[1],self.num_q,self.num_codes+1)
        return preds, codes

    def training_step(self,x, batch_idx):
        wav = x['wav'].unsqueeze(1)
        preds, targets = self(wav, x['prompt'])
        preds = preds.transpose(1,3)[:,:,:,:-1]
        loss = torch.nn.functional.cross_entropy(preds, targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self,x, batch_idx):
        wav = x['wav'].unsqueeze(1)
        preds, targets = self(wav, x['prompt'])
        preds = preds.transpose(1,3)[:,:,:,:-1]
        loss = torch.nn.functional.cross_entropy(preds, targets)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        opt = self.optimizer(self.trainer.model.parameters())

        if self.lr_scheduler is not None:
            if self.lr_scheduler.__name__ == 'SequentialLR':
                binds = gin.get_bindings('torch.optim.lr_scheduler.SequentialLR')
                lr_scheduler = self.lr_scheduler(opt, schedulers=[s(opt) for s in binds['schedulers']])
            else:
                lr_scheduler = self.lr_scheduler(opt) if self.lr_scheduler is not None else None
        else:
            lr_scheduler = None
        del self.optimizer
        del self.lr_scheduler
        opt_config = {'optimizer': opt}
        if lr_scheduler is not None:
            opt_config['lr_scheduler'] = {'scheduler': lr_scheduler,
                                          'interval': 'step',
                                          'frequency': 1}
        return opt_config

```

The full code for training can be found [here](https://github.com/mrpep/encodecgpt).

## Decoding audio
For the initial experiments, I trained the language model on [NSynth](https://magenta.tensorflow.org/nsynth), which is a
dataset with many synth samples spanning different music instruments and notes. All the samples are 4 seconds long and quite standarized. This makes it an ideal
dataset for first toy experiments.

### Autoregressive sampling
Once the model is trained, we want to generate audios with it. To sample from language models we have to do what is called autoregressive sampling.
- First, the prompt is fed to GPT2. An output is generated: this is the first sample generated by our model.
- We feed the prompt and the first sample generated by GPT2 and get the second sample.
- We keep repeating this process as long as we want, or until a special token (end of audio) is returned by the model. For this experiment we didn't use an end of audio token as
all the samples are 4 seconds long, so it is expected that after 4 seconds the model will return silence.

A nice thing about HuggingFace GPT2 implementation is that it allows key-value caching. This means that we don't need to generate all the intermediate outputs and compute attention with all the queries all the time during inference,
as intermediate results are cached at each autoregressive sampling iteration. This saves a lot of computations reducing the time required to generate the sequences.
This is not a minor detail as autoregressive sampling is expensive because it cannot be parallelized.

### Different flavors of sampling
One very important detail to discuss is how to pass from probability outputs to an actual output generated by the language model. There are many approaches, and those are 
discussed in more depth [here](https://huggingface.co/blog/how-to-generate). Some options are:
- **Greedy Search**: The output is the token with highest probability (argmax).
- **Sampling**: The output is picked randomly following the distribution returned by the model. By dividing the logits by a scalar called **temperature**, if we make it lower than 1
we can make the distribution sharper (give more probability to the most probable tokens) or if we make it higher than 1 the distribution gets more flat. In the extreme, a temperature tending to 0 will be equivalent to greedy search,
while a temperature tending to infinity will be like sampling from a uniform distribution. Temperature allows us to play with the generations, making them more diverse/random (higher temperature) or more stable (lower temperatures).
Sometimes, the generations might get stuck in loops and increasing temperature can be a solution.
- **Top-k Sampling**: Before sampling, we can choose the k tokens with highest probability and redistribute the mass probability among them.

In the following Python snippet we will use temperature sampling:

```python
def generate(prompt_filename, encodecmae, lm_model, temperature=1.0, generation_steps=300):
    prompt = encodecmae.extract_features_from_file(prompt_filename)
    prompt = prompt.mean(axis=0)
    with torch.no_grad():
        prompt = torch.from_numpy(prompt).to(lm_model.device)
        gpt_in = prompt.unsqueeze(0).unsqueeze(0)
        past_keys=None
        generation = []
        for i in tqdm(range(generation_steps)):
            outs = lm_model.gpt(inputs_embeds=gpt_in,past_key_values=past_keys)
            past_keys = outs['past_key_values']
            preds = lm_model.classification_head(outs['last_hidden_state'][0])
            preds = preds.view(preds.shape[0],lm_model.num_q,lm_model.num_codes+1)
            sampled_idxs = torch.cat([torch.multinomial(torch.nn.functional.softmax(preds[0,q,:]/temperature),1) for q in range(lm_model.num_q)])
            generation.append(sampled_idxs)
            in_idxs = torch.arange(lm_model.num_q, device=lm_model.device)*(lm_model.num_codes + 1) + sampled_idxs
            gpt_in = lm_model.vocab_embed(in_idxs).sum(axis=0).unsqueeze(0).unsqueeze(0)
        generation = torch.stack(generation)
        generation = roll_along(generation,-torch.arange(0,8,device=generation.device),0)
    
        audio = lm_model.encodec_model.decode([(torch.maximum(generation-1, torch.tensor(0, device=lm_model.device))[:-lm_model.num_q].T.unsqueeze(0),None)])
    audio = audio[0].cpu().detach().numpy()
    return audio
```

### The effect of temperature

Some examples of what happens when we move the temperature. This example is not in the training set of the model.

{{< music url="audio/t_prompt.wav" name="Prompt" artist="Unknown">}}
{{< music url="audio/t_001.wav" name="Temperature=0.01" artist="Unknown">}}
{{< music url="audio/t_01.wav" name="Temperature=0.1" artist="Unknown">}}
{{< music url="audio/t_03.wav" name="Temperature=0.3" artist="Unknown">}}
{{< music url="audio/t_07.wav" name="Temperature=0.7" artist="Unknown">}}
{{< music url="audio/t1.wav" name="Temperature=1.0" artist="Unknown">}}
{{< music url="audio/t15.wav" name="Temperature=1.5" artist="Unknown">}}
{{< music url="audio/t20.wav" name="Temperature=2.0" artist="Unknown">}}

We can hear that the generated samples resemble the prompt. However, when the temperature is too low (0.01 and 0.1), artifacts resulting
from outputs looping between tokens can be heard. This signals us that greedy search might be a bad idea. Increasing temperature leads to more organic results,
however more noise is also added. When the temperature is too high (>1.0), the generated samples start to sound random and very different from the prompt.

### Non NSynth sounds

What happens if we use as prompt something that is not a synth sound? Let's find out:

#### Prompt 1
{{< music url="audio/drill_prompt.wav" name="Drill prompt" artist="Unknown">}}
{{< music url="audio/drill_gen_t07.wav" name="Generated with T=0.7" artist="Unknown">}}

#### Prompt 2
{{< music url="audio/prompt3.wav" name="Prompt 2" artist="Unknown">}}

If we generate multiple times, we will get different results because of the random sampling. Listen to these 2 different outcomes:
{{< music url="audio/gen3_1.wav" name="Generated with T=0.7" artist="Unknown">}}
{{< music url="audio/gen3_2.wav" name="Generated with T=0.7" artist="Unknown">}}

#### Prompt 3
{{< music url="audio/prompt4.wav" name="Prompt 3" artist="Unknown">}}

EnCodecMAE seems to be capturing the periodicities of the prompt, in spite of the mean pooling over time:
{{< music url="audio/gen4_1.wav" name="Generated with T=0.7" artist="Unknown">}}
{{< music url="audio/gen4_2.wav" name="Generated with T=0.7" artist="Unknown">}}

#### MooSynth
{{< music url="audio/prompt5.wav" name="Prompt 3" artist="Unknown">}}

We can turn a Moooo into jurassic sounds:
{{< music url="audio/gen5_1.wav" name="Generated with T=0.7" artist="Unknown">}}
{{< music url="audio/gen5_2.wav" name="Generated with T=0.7" artist="Unknown">}}
{{< music url="audio/gen5_3.wav" name="Generated with T=0.7" artist="Unknown">}}

### Interpolation
I have also experimented interpolating 2 prompts. There are some problems if we want to do continuous interpolation:
- When the input is around 3 seconds long, because of NSynth samples, the generated outputs will start to fade out.
This can be partially fixed by generating until some point, let's say 2 seconds, and then leaving the sequence length fixed and
performing a First In First Out strategy.
- Also, when interpolating, the prompt is changing faster than the previous predicted tokens. This produces a mismatch; the first generated samples
corresponded to a prompt different from the present one.

In the examples below, instead of doing continuous interpolation, I generated 15 audios going from the Prompt 1 to the 2 in a linear interpolation, and then I concatenated them.

#### Example 1
{{< music url="audio/interp2_p1.wav" name="Prompt 1 (Mridangam)" artist="Unknown">}}
{{< music url="audio/interp2_p2.wav" name="Prompt 2 (NSynth Bass electronic)" artist="Unknown">}}
{{< music url="audio/interp2.wav" name="Linear interpolation" artist="Unknown">}}

#### Example 2
{{< music url="audio/interp3_p1.wav" name="Prompt 1 (NSynth mallet)" artist="Unknown">}}
{{< music url="audio/interp3_p2.wav" name="Prompt 2 (NSynth brass)" artist="Unknown">}}
{{< music url="audio/interp3.wav" name="Linear interpolation" artist="Unknown">}}

## References 

- Copet, J., Kreuk, F., Gat, I., Remez, T., Kant, D., Synnaeve, G., ... & Défossez, A. (2023). Simple and Controllable Music Generation. arXiv preprint arXiv:2306.05284.
- Défossez, A., Copet, J., Synnaeve, G., & Adi, Y. (2022). High fidelity neural audio compression. arXiv preprint arXiv:2210.13438.
- Oord, A. V. D., Dieleman, S., Zen, H., Simonyan, K., Vinyals, O., Graves, A., ... & Kavukcuoglu, K. (2016). Wavenet: A generative model for raw audio. arXiv preprint arXiv:1609.03499.
- Pepino, L., Riera, P., Ferrer, L. (2023). EnCodecMAE: Leveraging neural codecs for universal audio representation learning. arXiv preprint arXiv:2309.07391
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. OpenAI blog, 1(8), 9.
- Wang, C., Chen, S., Wu, Y., Zhang, Z., Zhou, L., Liu, S., ... & Wei, F. (2023). Neural codec language models are zero-shot text to speech synthesizers. arXiv preprint arXiv:2301.02111.