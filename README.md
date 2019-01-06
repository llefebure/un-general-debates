UN General Debates Analysis
===========================

This repo is a collection of experiments on historical speeches made at the UN General Debate, an annual forum for world leaders to discuss issues affecting the international community. These speeches form a historical record of these issues and represent an interesting corpus of text for analyzing this narrative over time.

In particular, I implement and apply various NLP methods for topic modelling, discovery, and interpretation, as well as studying how to model changes in topics over time.

The full text of these speeches was compiled and cleaned by researchers in the UK and Ireland, who used this data to study the position of different countries on various policy dimensions. See more info [here](https://arxiv.org/pdf/1707.02774.pdf).

## Setup

The structure of this project is based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. If you would like to run or extend this code yourself, follow the steps below for setting up a local development environment.

### Environment

Create the environment with:

```
make create_environment
```

This will create a Python3 `virtualenv` or `conda` environment (if `conda` is available). Next, activate the environment. With `conda` (what I'm using) this is:

```
source activate un-general-debates
```

Finally, install the requirements with the following command. Note that this uses `pip` behind the scenes. There are some packages that are not easily pip installable, so this command won't install everything. Look at the commented section in `requirements.txt` for packages that you should install manually.
```
make requirements
```

### Data

There are two raw data files that are used in this project. They are:
* [General Debates](https://www.kaggle.com/unitednations/un-general-debates): This has all of the raw text and metadata from the speeches.
* [Country Code Mapping](https://www.kaggle.com/juanumusic/countries-iso-codes): This has a mapping from ISO Alpha-3 country codes to country names.

This project also uses pretrained Wikipedia2vec vectors for word and entity (Wikipedia page) embeddings. See [here](https://wikipedia2vec.github.io/wikipedia2vec/) for more details.

The following will download all of the raw data and preprocess the appropriate files for you. If you haven't used the Kaggle API before, some additional setup will be required for this to work.

```
make data
```

Preprocessed data files are written to `data/processed/`.

## Methods

### Paragraph Tokenization

A key observation in this dataset is that each of these speeches consists of discussion on a multitude of topics. If every speech contains discussion on poverty and terrorism, a topic model such as LDA trained on entire speeches as documents, for example, will have no way of understanding that terms like "poverty" and "terrorism" should be representative of different topics.

To counter this problem, I tokenize each speech into paragraphs and treat each paragraph as a separate document for analysis. A simple rule based approach that looks for sentences separated by a newline character performs reasonably well on the task of paragraph tokenization for this dataset. After this step, the number of documents jumps from around 7,500 (full speeches) to nearly 300,000 (paragraphs).

### Topic Modelling

#### LDA

Applying LDA with `gensim` and visualizing resulting topics with `pyLDAvis` reveals some easily interpretable topics such as nuclear weapons, Africa, and Israel/Palestine. See [notebooks/LDA.ipynb](notebooks/LDA.ipynb).

#### Dynamic Topic Modelling

A Dynamic Topic Model [1] is basically an extension of LDA to allow topic representations to evolve over fixed time intervals such as years. I wrote about applying this method [here](https://towardsdatascience.com/exploring-the-un-general-debates-with-dynamic-topic-models-72dc0e307696). As an example, the model learned a topic about "Human Rights", and for this topic, a plot of probabilities over time for selected terms is shown below. Note the rising use of "woman" and "gender", the decline of "man", and the inverse relationship between "mankind" and "humankind".

![Human Rights Topic Probabilities](reports/figures/humanrights.png)

This code uses `gensim`'s wrapper to the original C++ implementation to train DTMs. See the [docs](https://radimrehurek.com/gensim/models/wrappers/dtmmodel.html) for instructions on setup. You will need to either download a precompiled binary or build one manually.

To train a DTM on this dataset, refer to [src/models/dtm.py](src/models/dtm.py). Note that the inference takes quite a while: almost 8 hours for me on a n1-standard-2 (2 vCPUs, 7.5 GB memory) instance on Google Cloud Platform. The script will save the model and a copy of the processed data into `models/`, and you can use the notebook [notebooks/DTM.ipynb](notebooks/DTM.ipynb) to explore the learned topics.

#### Topic Labelling

The topic models above represent topics as distributions over words. Interpreting learned topics through lists of individual words has a high cognitive overhead, so several methods for labelling topics in a more human friendly way have been developed such as [2] and [3]. These methods leverage utilize chunking/noun phrases, frequency statistics, Wikipedia titles/concepts, and more to generate descriptive phrases to describe topics.

Taking inspiration from [2] especially, I develop a method for labelling topics leveraging pretrained word and entity (Wikipedia title) embeddings from Wikipedia2vec [4]. Specifically, my method consists of the following steps to label a given topic:

1) For each document assigned to the topic, extract all noun chunks and match each to a corresponding Wikipedia entity (if one exists). Take the top N entities by frequency as label candidates for the next step.

2) For each label candidate from 1), compute the mean similarity between its embedding and that of each word in the term list that represents the topic (top M terms by marginal probability). Rank the label candidates by this score.

3) Represent the topic with the top 3 label candidates as ranked in 2).

An example of output from this method:
* **Top Terms:** nuclear, weapon, non, treaty, arm, ban, proliferation, moon, use, mass
* **Entity Labels:** Nuclear proliferation, Nuclear weapon, Disarmament

### Semantic Hashing

In [6], the authors present an interesting method for hashing documents using a deep generative model. I implemented the unsupervised version of the model that uses a VAE to encode a TFIDF vector and decode into a softmax distribution over the vocabulary. This could be used as a preprocessing step to bucket documents before applying more expensive pairwise comparison methods on documents within buckets.

## References

[1] [Dynamic Topic Models](https://mimno.infosci.cornell.edu/info6150/readings/dynamic_topic_models.pdf)

[2] [Automatic Labelling of Topics with Neural Embeddings](http://www.aclweb.org/anthology/C16-1091)

[3] [Automatic Labeling of Multinomial Topic Models](http://sifaka.cs.uiuc.edu/czhai/pub/kdd07-label.pdf)

[4] [Wikipedia2Vec: An Optimized Tool for Learning Embeddings of Words and Entities from Wikipedia](https://arxiv.org/pdf/1812.06280.pdf)

[5] [LDAvis: A method for visualizing and interpreting topics](https://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf)

[6] [Variational Deep Semantic Hashing for Text Documents](https://arxiv.org/pdf/1708.03436.pdf)
