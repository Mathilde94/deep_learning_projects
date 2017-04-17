# First Exploration on Machine Learning Models with Tensorflow

This repository contains a set of machine learning models I learnt and tried recently
after getting the Coursera Machine Learning Certificate with Andrew Ng (great course!)
and the Deep Learning Udacity course (https://www.class-central.com/mooc/5681/udacity-deep-learning,
pretty brief but requires you to search and read about the algorithms).

I also wanted to learn a bit more about Tensorflow and why it is getting more popular.
This library is a first hands on it.

This WIP library using Tensorflow can help building:

 * logistic regression models           (trained on notMnist data)

 * neural network models                (trained on notMnist data)

 * convolutional neural network models  (trained on notMnist data)

 * skip gram models for text            (trained on text8)

 * long short term memory models        (trained on text8)


## Getting started

Thanks to a dummy `main.py` and configurations easily editable, you can run those like this:

```
python main.py <train_logistic|train_nn|train_nn_conv|train_texttrain_text_lstm>
```

and see the corresponding model getting trained and learning.
Below is how things are happening:

### The Model
The Model class has:

 * a configuration: set of parameters relative to the model. For instance, the number
 and size of `hidden|convolutional|fully_connected` layers, a name, etc

 * a set of hyper parameters for that model that will be passed to a generic Trainer Model (less
 Trainer instances that model as the logistic, neural network and convolutional networks can share
 a same type of Trainer with Tensorflow in this library) for its learning.

 * methods to populate specific TensorFlow variables for its model (`train|valid|test` placeholders,
 as well as Tensorflow variables for its parameters)

 * method to `feed_forward` each input

You have 5 main models inheriting the general `MLModel`: `LogisticRegression`, `NeuralNetwork`,
`ConvolutionNeuralNetwork`, `SkipGram`,  `LSTM`.

Example for the Neural Network Model:

```
class NeuralNetwork(MLModel):

    configuration = NeuralNetworkConfiguration  # contains layers and model configurations
    hyper_parameters = {
        'epochs': 301,
        'display_epochs': 50,
        'keep_prob': 0.9,
        'learning_rate': 0.5,
        'lambda_rate': 0.03,
    }

    def populate_model_variables(self):
        for layer in self.hidden_layers:
            l = LayerParameter()
            l.weights = tf.Variable(tf.random_normal([layer['input'], layer['output']]))
            l.biases = tf.Variable(tf.random_normal([layer['output']]))
            self.parameters.hidden_layers.append(l)

        # Same concept the final Layer etc...

    def feed_forward(self, input_data, keep_prob=1.0, _lambda=0.0):
        hidden = data
        for layer in self.parameters.hidden_layers:
            hidden = tf.add(tf.matmul(hidden, layer.weights), layer.biases)
            hidden = tf.nn.relu(hidden)
            hidden = tf.nn.dropout(hidden, keep_prob)

        layer = self.parameters.final_layer
        hidden = tf.add(tf.matmul(hidden, layer.weights), layer.biases)
        # Adding regularization costs (...)

        return output
```

The text Models have parts coming from the Udacity Exercises.


### The Trainer
The Trainer is the second most important object as it will run a model.
There are 3 `BasicTrainer`:

   * the main `Trainer` that can be used for logistic regressions, neural networks and
   convolutional networks

   * a `TextTrainer` for Skip Gram (TODO: needs to rename it)

   * an `LSTMTrainer` that runs as well a bit differently than the others for
    LSTM models

Each trainer gets it main hyper parameters (learning rate, keep prob, alpha rate, batch size etc)
from the model it has to run against.

The text Trainers come from the exercises from the Udacity Exercises, they have been
readapted to my model only, I want to create my own soon.


### The Classifier
The Classifier is an object that will prepare the model and run its trainer against it.
It will as well record statistics on the learning and show loss, accuracies,
text grouping etc depending on the Machine Learning Algorithm.

The Classifier can save a session and stats to a file and can start again from
a specific session (Tensorflow session).


Eventually, this is what each code looks like at the high level:

```
    model = LogisticRegression()
    classifier = Classifier(model)
    classifier.train(data_for_trainer)
    classifier.stats()

    nn = ConvolutionNeuralNetwork()
    classifier_nn = Classifier(nn)
    classifier_nn.train(data_for_trainer)
    classifier_nn.stats()
```

You can check the `ipynb` files that show a basic use for each of them, and some basic stats.

## Training the models

A few challenges and notes I can gather about each model training. Once I got the main code running and
training my algorithm, I would tune the model hyper parameters and configurations to get the best accuracies and
make sure it is learning and decreasing the loss.

 * Generally, having regularization always helped converging faster and avoiding overfitting on training set.

 * Using minibatch method is much faster and still gives good results

 * Sometimes, the algorithm would become very very slow to learn after a certain number of steps.
Hence I introduced a alpha rate that would change over time and got better speed of learning.

 * I also used the `keep_prob` Tensorflow option to drop some results, that helped with accuracy

I am using a MAC 10.10.8 and ran most of my trainings with around 20K training points for logisitc regression,
Neural Network and Convolutional Networks. But sometimes, I would use 60K to 150K datapoints for those algorithms
to test with my parameters.


## Sample Results on NotMNIST dataset

On the notMNIST dataset with Neural Networks, I reached **96%** of test accuracy after an hour of running
(it reached about 93.5% after 20minutes).

The Convolutional Network reaches a **95%** test accuracy in 12 minutes for the NotMNIST dataset.

Output Sample showing cost for training, valid and test loss and accuracies :
```
Step: 0: l=5.696619 l_valid=36.298420 l_test=35.981113 minibatch=12.5% valid=9.8% test=9.9%
Step: 50: l=1.236942 l_valid=1.128440 l_test=0.952458 minibatch=68.0% valid=67.9% test=74.8%
Step: 100: l=0.913848 l_valid=0.709012 l_test=0.490568 minibatch=75.0% valid=80.4% test=86.7%
Step: 150: l=0.717094 l_valid=0.624215 l_test=0.406925 minibatch=77.3% valid=81.7% test=88.0%
Step: 200: l=0.770429 l_valid=0.602519 l_test=0.393613 minibatch=78.1% valid=82.2% test=88.7%
Step: 250: l=0.589795 l_valid=0.570907 l_test=0.357846 minibatch=84.4% valid=82.8% test=90.0%
Step: 300: l=0.751026 l_valid=0.550192 l_test=0.331506 minibatch=75.0% valid=83.9% test=90.2%
Step: 350: l=0.718476 l_valid=0.534638 l_test=0.320542 minibatch=76.6% valid=84.0% test=90.8%
Step: 400: l=0.609484 l_valid=0.517977 l_test=0.302995 minibatch=82.0% valid=84.7% test=91.1%
Step: 450: l=0.501893 l_valid=0.501924 l_test=0.281929 minibatch=85.2% valid=85.0% test=91.5%
Step: 500: l=0.455414 l_valid=0.490821 l_test=0.271553 minibatch=86.7% valid=85.4% test=92.1%
...
Step: 2800: l=0.318769 l_valid=0.390600 l_test=0.175587 minibatch=89.8% valid=88.4% test=94.9%
Step: 2850: l=0.294269 l_valid=0.384566 l_test=0.172313 minibatch=90.6% valid=88.6% test=94.8%
Step: 2900: l=0.232945 l_valid=0.390779 l_test=0.171455 minibatch=91.4% valid=88.9% test=95.0%
```

![Alt text](/images/logistic_regression_stats.png?raw=true "Convolutional Neural Network")


For LSTM, it was funny to see how a random text can become more english:

First step gave this snippet:
```
nans  bcipu ied evuyogivsdafszhhjywf   gkaahebqjsy  cjn eitj dm u emee hnnas hin
jgo d  zirvxon qtetqa en gecjdax  iad tdldf gckgv xbnrxnwllqaneex hp htu  nnibta
edcpyweesanujlnhio tk lytptezj  hjihirtr ninmiarfbd ujmdsb zupe c pu oatr em kzn
qsce hethdeinshzkwol f ienwc yrgyut rstapiijamz   uceudvq efk cctzjasclvioiajens
fn ahsto  iilinqa ptf cv e usoegcnsei htbovfclz n y abet oyki lt egqv unzp  a ij
```
to become:
```
is docker including alfames roses ould to and stating is while on joll the ama t
 top is finally beroce contract wys both a septess was about orc pafer stitt fem
haardman sond about one nine two nine three octims gooder ween a port of fourth
le as five paired community distirlle x of king gaugglean that is notablg leanfi
d and totes in thet found number invuloted service seem arrian new plased junned
```


For SkipGram: some similar words examples after a few seconds learning are:
```
their: its, the, his, any, lossless, supposing, cracking, lou,
into: to, when, nicomachean, for, as, annihilate, acidosis, in,
to: into, for, from, would, gretzky, walther, under, disability,
and: or, of, in, s, references, but, which, at,
three: zero, eight, four, two, five, intangible, six, haddock,
during: same, at, copies, on, cleaved, from, pesticide, relay,
not: exonerated, t, will, they, guilford, which, tammy, nobody,
would: will, mughals, can, may, should, to, marv, who,
american: unpleasant, conformity, zangger, national, decrypt, fuck, offense, liberator,
people: specifies, vietnamese, rubens, villiers, glories, bush, update, foals,
is: was, has, are, profane, townships, as, by, be,
have: has, had, be, were, are, cosmonauts, alshehhi, within,
some: the, these, originally, many, koi, kangaroo, working, this,
with: in, by, through, for, of, on, from, cw,
```
![Alt text](/images/text_graph.png?raw=true "Text Graph")


A quick logistic regression training would give an accuracy of 84%. The graph belows
show the loss decreasing, the training, validation and test accuracies:
![Alt text](/images/logistic_regression_stats.png?raw=true "Logistic Regression")


## Next Steps
Once I have some time, I would love to try the Image Classification on CIR Images!


## Articles Read
 * How I learnt about the Convolutional Networks: http://cs231n.github.io/convolutional-networks/
 * How I learnt about LSTM: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
 * How I learnt about Tensorflow: https://www.tensorflow.org/

Note: This is a WIP library until I can have some time to clean and improve it :).

