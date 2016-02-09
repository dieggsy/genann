#GENANN

GENANN is a very minimal library for training and using feedforward artificial neural
networks (ANN) in C. Its primary focus is on being simple, fast, and hackable. It achieves
this by providing only the necessary functions and little extra.

##Features

- **ANSI C with no dependencies**.
- Contained in a single source code and header file.
- Simple.
- Fast and thread-safe.
- Easily extendible.
- Implements backpropagation training.
- Compatible with training by alternative methods (classic optimization, genetic algorithms, etc)
- Includes examples and test suite.
- Released under the zlib license - free for nearly any use.

##Example Code

Four example programs are included.

- `example1.c` - Trains an ANN on the XOR function using backpropagation.
- `example2.c` - Trains an ANN on the XOR function using random search.
- `example3.c` - Loads and runs an ANN from a file.
- `example4.c` - Trains an ANN on the [IRIS data-set](https://archive.ics.uci.edu/ml/datasets/Iris) using backpropagation.

##Quick Example

Here we create an ANN, train it on a set of labeled data using backpropagation,
ask it to predict on a test data point, and then free it:

```C
#include "genann.h"

/* New network with 5 inputs,
 * 2 hidden layer of 10 neurons each,
 * and 1 output. */
GENANN *ann = genann_init(5, 2, 10, 1);

/* Learn on the training set. */
for (i = 0; i < 300; ++i) {
    for (j = 0; j < 100; ++j)
        genann_train(ann, training_data_input[j], training_data_output[j], 0.1);
}

/* Run the network and see what it predicts. */
printf("Output for the first test data point is: %f\n", *genann_run(ann, test_data_input[0]));

genann_free(ann);
```

Not that this example is to show API usage, it is not showing good machine
learning techniques. In a real application you would likely want to learn on
the test data in a random order. You would also want to monitor the learning to
prevent over-fitting.


##Usage

###Creating and Freeing ANNs
```C
GENANN *genann_init(int inputs, int hidden_layers, int hidden, int outputs);
GENANN *genann_copy(GENANN const *ann);
void genann_free(GENANN *ann);
```

Creating a new ANN is done with the `genann_init()` function. It's arguments
are the number of inputs, the number of hidden layers, the number of neurons in
each hidden layer, and the number of outputs. It returns a `GENANN` struct pointer.

Calling `genann_copy()` will create a deep-copy of an existing GENANN struct.

Call `genann_free()` when you're finished with an ANN returned by `genann_init()`.


###Training ANNs
```C
void genann_train(GENANN const *ann, double const *inputs, double const *desired_outputs, double learning_rate);
```

`genann_train()` will preform one update using standard backpropogation. It
should be called by passing in an array of inputs, an array of expected output,
and a learning rate. See *example1.c* for an example of learning with
backpropogation.

A primary design goal of GENANN was to store all the network weights in one
contigious block of memory. This makes it easy and efficient to train the
network weights directly using direct-search numeric optimizion algorthims,
such as [Hill Climbing](https://en.wikipedia.org/wiki/Hill_climbing),
[the Genetic Algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm), [Simulated
Annealing](https://en.wikipedia.org/wiki/Simulated_annealing), etc.
These methods can be used by searching on the ANN's weights directly.
Every `GENANN` struct contains the members `int total_weights;` and
`double *weight;`.  `*weight` points to an array of `total_weights`
size which contains all weights used by the ANN. See *example2.c* for
an example of training using random hill climbing search.

###Saving and Loading ANNs

```C
GENANN *genann_read(FILE *in);
void genann_write(GENANN const *ann, FILE *out);
```

GENANN provides the `genann_read()` and `genann_write()` functions for loading or saving an ANN in a text-based format.

###Evaluating

```C
double const *genann_run(GENANN const *ann, double const *inputs);
```

Call `genann_run()` on a trained ANN to run a feed-forward pass on a given set of inputs. `genann_run()`
will provide a pointer to the array of predicted outputs (of `ann->outputs` length).

##Extra Resources

The [comp.ai.neural-nets
FAQ](http://www.faqs.org/faqs/ai-faq/neural-nets/part1/) is an excellent
resource for an introduction to artificial neural networks.

If you're looking for a heavier, more opinionated neural network library in C,
I highly recommend the [FANN library](http://leenissen.dk/fann/wp/). Another
good library is Peter van Rossum's [Lightweight Neural
Network](http://lwneuralnet.sourceforge.net/), which despite its name, is
heavier and has more features than GENANN.

##Hints

- All functions start with `genann_`.
- The code is simple. Dig in and change things.
