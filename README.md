# simple-sentence-analyzer
This repository is for me (and maybe for you) to understand a very simple NLP task: detecting if a given sentence is positive or negative. GloVe's pretrained word vectors are used. Both NumPy (main.py) and Keras (main2.py) are used. This has a very small training set, but pretrained vectors are sufficient to get a decent accuracy. GloVe vectors have several dimensions. More dimension means more accuracy in this case. Download GloVe from [here](https://nlp.stanford.edu/projects/glove/) (glove.6B.zip) and put it into data folder. I also put python file (mistakes.py) to see the examples from the test set that model misses. For this and your further Keras applications, I recommend using [one click deep learning](https://www.crestle.ai/billing-usage).


[![HitCount](http://hits.dwyl.io/kbulutozler/simple-sentence-analyzer.svg)](http://hits.dwyl.io/kbulutozler/simple-sentence-analyzer)

