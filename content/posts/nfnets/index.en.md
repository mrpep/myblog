---
title: "Normalizer free networks"
tags: ["resnets","normalization","deep learning"]
date: 2023-04-15T01:13:43-03:00
draft: true
math: true
raw: true
---

#### Signal propagation plots (SPP)

Plotting the statistics of hidden activations can be benefical to gain insights of possible bugs in deep learning
models implementations, problems with initialization and activations.
The authors of the paper propose giving the network inputs sampled from a normal distribution and then plotting the following
statistics:
- Average Channel Squared Mean
- Average Channel Variance

The authors calculate these metrics at the output of residual blocks in a ResNet model. They also calculate the average
channel variance at the output of the residual branch before the skip connection.

They find that ResNets behave...

#### Normalizing Free Networks Design

The authors introduce 2 scalars to scale the residual branch input and outputs. 
The input is scaled by {{< raw >}}\(\frac{1}{\beta}\){{< /raw >}}, and the output by {{< raw >}}\(\alpha\){{< /raw >}}
{{< raw >}}
\[ x_{l+i} = x_{l} + \alpha*f(\frac{x_l}{\beta_i})\]
{{< /raw >}}
{{< raw >}}\(\beta_i\){{< /raw >}} is defined such that the input to the residual branch has unit variance, so its value is:
{{< raw >}}
\[ \beta_i = \sqrt{Var(x_l)} = \sqrt{Var(x_{l-1}) + \alpha^2}\]
{{< /raw >}}

{{< raw >}}\(\alpha\){{< /raw >}} is set to 0.2 and defines the growth rate of the variance across residual blocks.

At every transition block, variance is reset as seen in the SPP, so at the beginning of each residual stage, the value of
{{< raw >}}\(\beta\){{< /raw >}} is reset to {{< raw >}}\(\sqrt{1+\alpha^2}\){{< /raw >}}.

Another effect that has to be taken into account is that the activations are not centered in zero, so the mean activation
will be larger than 0 and grow at each residual block. To avoid the mean shift and the variance being less than 1 because of the activations,
authors propose scaled weight standarization:

{{< raw >}}
\[ \hat{W}_{i,j} = \gamma\frac{W_{i,j} - \mu_{W_i}}{\sigma_{W_i}\sqrt{N}}\]
{{< /raw >}}
