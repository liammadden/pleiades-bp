To install dependencies and run the experiment, use the following command:

```
uv run pleiades\run.py
```

In pleiades\run.py, you can change the sequence length, number of layers, number of heads, embedding dimension, head dimension, MLP dimension, number of samples, batch size, number of epochs, and step size. A data set from TinyStories (https://arxiv.org/abs/2305.07759) is processed using pleiades/data.py, a Transformer model is created using pleiades/model.py, the model is trained with Adam using pleiades/train.py, and the loss evolution is plotted using pleiades/plotting.py.