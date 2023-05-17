---
title: "My approach to ML pipelines"
tags: ["pipeline", "gin", "python", "machine learning", "dev"]
date: 2023-04-15T01:13:43-03:00
draft: true
math: true
raw: true
---

In this article I'll show you the way I run machine learning experiments. It is not perfect, it is not for every use case,
and it's just what I have come up after several years of doing research in deep learning.
This article is focused mostly on the organization of ML experiments and the code to produce this particular setting. However,
in order to make it less abstract, an end to end example of training an audio model for acoustic event detection is shown.
My setup relies on several popular and not so popular libraries, mainly: Gin, Pandas and Pytorch Lightning, so I'll talk a lot about these libraries and how they can be used and combined.
So, let's move on and start talking about pipelines.

### A machine learning pipeline

Usually, my experiments are structured as in the diagram below: 

{{< mermaid >}}
flowchart LR
    id1[(Dataset)]
    id2[Load Dataset]
    id3[Split Dataset]
    id4[Extract Features]
    id5[Train Model]
    id6[Evaluate Model]
    id1 --> id2
    id2 --> id3
    id3 --> id4
    id4 --> id5
    id5 --> id6
    id6 --> id4
{{< /mermaid >}}

I'll take some lines to explain each node in the diagram:
* **Dataset**: this is the fuel of any machine learning model. Without data there is nothing, so it's a must node.
Also, it is probably the most diverse node, as a dataset could be something as small as [MNIST](), which can be fully loaded into RAM,
or as huge as [The Pile](https://pile.eleuther.ai/) or [VoxPopuli](https://github.com/facebookresearch/voxpopuli), which won't fit into RAM in most computers. 
To give you an idea, VoxPopuli is 6.4 terabytes of data! Moreover, the data is in a compressed audio format, so to be usable by a model, it probably has to be
decompressed, making the required space even larger.
A dataset could be made up of images, audio, video, tabular data, text, voxels, and a very long etc.
Even if a dataset is just images, it could be in different image formats, or maybe if the dataset authors don't want
to release the images themselves, it could be features representing the original images in a lossy way.
Also, the images could be faces, cars, landscapes... And they could be in high resolution like 1024x1024, or just 28x28 like MNIST.
So, datasets are very diverse, and every dataset will require a different setup to load it and work with it.
* **Load Dataset**: as explained above, every dataset will need special code to turn it into something usable by our models.
There are 2 possible approaches in my setup:
    1) Dataset is small and can fit into the RAM. In this case, I could just open every file of the dataset, ie. every jpeg image, and store it into an array that will be further processed.
Alongside the images themselves, I can store the labels. This is an ideal scenario as reading from disk, which is slow, will happen only once.
    2) Dataset is large and can't fit into the RAM. This is what I am more used to, as I work in audio, a single second of audio can be 16000 samples at 16 bits, so datasets rapidly fill the RAM memory.
In this case, instead of loading the dataset, what I do is to store the metadata in a Pandas Dataframe. The most important bits of metadata are the file path, and the labels. But other metadata are useful,
such as duration, format, speaker id, etc... So, in this case, the data is actually loaded during the model training, when each batch is created.
* **Split Dataset**: when working with ML models, it is very important to split our data in subsets, usually train, validation and test.
There are many different strategies to split the data, such as k-fold cross validation and leave one sample or group out. So, this step will vary depending on
the used dataset, if there exists an standard way of splitting and working with that dataset, etc...
* **Extract features**: many times the data is not used as it is by the model, but features are extracted from it. Again, there are 2 scenarios here:
    1) Feature extraction takes a long time and the extacted features don't occupy too much disk space. In this case, features can be calculated once and kept in the RAM or saved to the disk.
    2) Feature extraction is fast. If feature extraction is fast, it can be done during training when each batch is created and it won't be a bottleneck, specially if multiprocessing is used.
* **Train model**: This is usually the step that takes the most time. Training can last minutes, hours, days, weeks, months, ... maybe years if you forget to stop your experiments.
 So you don't want to lose your model if the system fails after 3 days of having started training. We have to implement checkpointing. Also, we want it to be easy to restart the experiment in these failure cases.
 Moreover, we want the model to be easily customizable, as we will probably do some hyperparameters tuning.
* **Evaluate model**: We want to see how the model is performing on the task of interest. The evaluation will happen during the training, because we want to be able to monitor the training and validation metrics while fitting our models.
To do this, we will use loggers. Also, once the model is trained, we want to run unseen examples and check the performance of the model on them.
The model evaluation will give us feedback and according to that feedback we will return to a previous step and repeat.
In the diagram, the return is to the feature extraction step, so maybe we can try extracting different features and see if it improves performance.
But it's just an example, we could return to the dataset and search for better data, or preprocess it, or we could just return to the model training and change some hyperparameters.


### Configuration files with Gin
<div style="text-align: center;">
<img src="images/gin_dalle.png" style="max-width: 50%;">
</div>

So, we will need to change things while iterating models. We will change the batch size, the number of layers, 
details of the feature extraction, etc... etc.... A naive option to do this would be to copy the code and change only the desired values.
But you would be replicating a lot of code in a lot of folders, and it doesn't sound elegant. Moreover, imagine you discover a bug in your code. 
Now you have to copy paste all the codes and create folders again. Tedious... There has to be a better way.

Let's talk about [Gin](https://github.com/google/gin-config). Gin is a configuration framework for Python, which is well suited for ML experiments.

Let's make a minimalistic example of using Gin, and then expand over it. This will be in the configuration file 'config.gin':

```python

FSD50K_PATH = '/datasets/fsd50k'

execute_pipeline:
    tasks = [@set_seed,
             @load_dataset,
             @split_dataset,
             @extract_features,
             @fit_model]

set_seed.seed = 42
load_dataset:
    dataset = 'fsd50k'
    dataset_path = %FSD50K_PATH
split_dataset:
    type = 'train_val_test'
    proportion = [70,10,20]
    mode = 'random'
fit_model:
    model = @Resnet()
Resnet:
    n_layers = 10

```

Then we will make a test_minimal.py with the following content:

```python
from loguru import logger
import gin

@gin.configurable
def execute_pipeline(tasks):
    for t in tasks:
        logger.info(f'Executing task {t.__name__}')
        t()

@gin.configurable
def set_seed(seed):
    logger.info(f'Running experiment with seed {seed}')

@gin.configurable
def load_dataset(dataset, dataset_path):
    logger.info(f'Loading {dataset} located in {dataset_path}')

@gin.configurable
def split_dataset(type, proportion, mode):
    logger.info(f'Splitting dataset by {type} with proportion {proportion} and {mode} mode')

@gin.configurable
def extract_features(type):
    logger.info(f'Extracting {type}')

@gin.configurable
def fit_model(model):
    logger.info(f'Fitting model {model}')

@gin.configurable
class Resnet():
    def __init__(self, n_layers=5, n_classes=20):
        logger.info(f'Creating Resnet model with {n_layers} layers to classify {n_classes} classes')

def main():
    gin.parse_config_file('minimal_example.gin')
    logger.info('\nExperiment configuration: \n'+gin.config_str())
    
    execute_pipeline()

main()
```

If we run
```bash
python3 minimal_example.py
```
The following output will be generated:
```bash
2023-04-26 15:24:36.316 | INFO     | __main__:main:37 - 
Experiment configuration: 
# Macros:
# ==============================================================================
FSD50K_PATH = '/datasets/fsd50k'

# Parameters for execute_pipeline:
# ==============================================================================
execute_pipeline.tasks = \
    [@set_seed, @load_dataset, @split_dataset, @extract_features, @fit_model]

# Parameters for extract_features:
# ==============================================================================
extract_features.type = 'melspectrogram'

# Parameters for fit_model:
# ==============================================================================
fit_model.model = @Resnet()

# Parameters for load_dataset:
# ==============================================================================
load_dataset.dataset = 'fsd50k'
load_dataset.dataset_path = %FSD50K_PATH

# Parameters for Resnet:
# ==============================================================================
Resnet.n_layers = 10

# Parameters for set_seed:
# ==============================================================================
set_seed.seed = 42

# Parameters for split_dataset:
# ==============================================================================
split_dataset.mode = 'random'
split_dataset.proportion = [70, 10, 20]
split_dataset.type = 'train_val_test'

2023-04-26 15:24:36.316 | INFO     | __main__:execute_pipeline:7 - Executing task set_seed
2023-04-26 15:24:36.316 | INFO     | __main__:set_seed:12 - Running experiment with seed 42
2023-04-26 15:24:36.316 | INFO     | __main__:execute_pipeline:7 - Executing task load_dataset
2023-04-26 15:24:36.316 | INFO     | __main__:load_dataset:16 - Loading fsd50k located in /datasets/fsd50k
2023-04-26 15:24:36.317 | INFO     | __main__:execute_pipeline:7 - Executing task split_dataset
2023-04-26 15:24:36.317 | INFO     | __main__:split_dataset:20 - Splitting dataset by train_val_test with proportion [70, 10, 20] and random mode
2023-04-26 15:24:36.317 | INFO     | __main__:execute_pipeline:7 - Executing task extract_features
2023-04-26 15:24:36.317 | INFO     | __main__:extract_features:24 - Extracting melspectrogram
2023-04-26 15:24:36.317 | INFO     | __main__:execute_pipeline:7 - Executing task fit_model
2023-04-26 15:24:36.317 | INFO     | __main__:__init__:33 - Creating Resnet model with 10 layers to classify 20 classes
2023-04-26 15:24:36.317 | INFO     | __main__:fit_model:28 - Fitting model <__main__.Resnet object at 0x7f4b11c63110>
```

Let's explain what is happening:

The Python code in test_minimal.py defines some dummy functions, that by the moment won't do anything, but are useful to test the project structure.

The `execute_pipeline()` function will iterate over tasks, which is a list of functions and execute them.
We can see that there is a `t()` and so, every function is called without any arguments. 
That's where gin does its magic. 

When we do something like `set_seed.seed=42` in the gin configuration file, this is setting
42 as the default value for the `seed` arg in the `set_seed()` function. So, now when calling `set_seed()`, 
it is no longer necessary to pass a value for `seed`, as the default value supplied by gin will be used.
If we take a look at the gin configuration file we will see other symbols:
- The `@` allows us to refer to functions or classes defined in our code. To be used from the gin config, we have to decorate
those functions with `@gin.configurable`.
For example `@set_seed` sets the function `set_seed()` as a default value in the list of tasks that are used as arg to the `execute_pipeline()` function.
A slightly different example is the use of `@Resnet()`. Adding the `()` will instantiate the Resnet class, so instead of receiving the class as argument,
an instance of the class will be received. The gin documentation says that we should avoid using this feature, however, sometimes it is practical.

- The `%` is used to refer to **Macros**. Macros are constants defined in the gin config, like for example `FSD50K_PATH` in the first line. 
We can use those macros in multiple places of our config by writing, for example, `%FSD50K_PATH`.

- We will see some more advanced gin features, like **scopes**, in next sections.

Finally, in the test_minimal.py `main()`, we load the config with `gin.parse_config_file`, and we can print it with a nice format by calling `gin.config_str()`.

### Moving further with GinPipe

<div style="text-align: center;">
<img src="images/ginpipe.jpg" style="max-width: 50%;">
</div>

So, it seems like **Gin** is a nice way to configure our experiments and pipelines. However, there are some limitations:

- Any function we want to use from gin needs to be decorated with `@gin.configurable`. If we want to use external libraries, like for example, scikit-learn, we should manually make
each function configurable. This can be done in gin by using the `gin.config.external_configurable` function applied to each function/class we want to use.

- We want each task to pass data to the other ones. For example, once we load and process the dataset, the resulting object has to be accesible for the next tasks.

- We want to be able to run our experiments from a terminal by passing the configuration file/s, and we want to be able to modify specific variables from the configuration file.
For example, we might have a configuration file defining a model, and then we want to run the same experiment but only change the number of layers.
One way would be to make a copy of the configuration file and change the value of that parameter. A nicer option would be to call the command from terminal, using the same configuration file,
and change that parameter from the command itself, by supplying a ``mod`` argument.

- We want to be able to compose multiple configuration files in a modular way. For example, we can have a configuration file for our dataset, another for the model, and a main one.
Then, if we want to try different models, we can have configuration files for each model, and just replace the model configuration file when calling the command.

To enhance gin, and make it more suitable for executing ML pipelines by solving the outlined limitations, I created [GinPipe](https://github.com/mrpep/ginpipe)



Let's start building our example and explaining the GinPipe features as they appear.

### Reproducibility

<div style="text-align: center;">
<img src="images/seed_leonardo.jpg" style="max-width: 50%;">
</div>

The first thing 

### Reading and splitting the dataset

<div style="text-align: center;">
<img src="images/panda_leonardo.jpg" style="max-width: 50%;">
</div>

### Creating batches with Pytorch

<div style="text-align: center;">
<img src="images/batches_leonardo.jpg" style="max-width: 50%;">
</div>

### Powering up the model with Pytorch Lightning

<div style="text-align: center;">
<img src="images/lightning_leonardo.jpg" style="max-width: 50%;">
</div>


### Torchmetrics and evaluation

<div style="text-align: center;">
<img src="images/metrics_leonardo.jpg" style="max-width: 50%;">
</div>

### Creating a demo with Gradio

### Putting everything inside a container