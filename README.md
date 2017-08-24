# Mining_The_Social_Web
Acquire, analyze, and summarize data from all corners of the social web.


The repository has three parts:

  1. Scraper, to obtain data from social media.
  
  2. Training script, to train a classifier.
  
  3. A script to classify text using the previous trained classifier.
  
  
#### Required libraries

To use the classifier, the following libraries must be installed:

i. [scikit-learn==0.18.2](http://scikit-learn.org/stable/install.html)

ii. [nltk==3.2.4](https://pypi.python.org/pypi/nltk)

iii. [numpy==1.13.1](http://www.numpy.org/)

iv. [pandas==0.20.3](http://pandas.pydata.org/)

v. [scipy==0.19.1](https://www.scipy.org/) 

vi. [twitter==1.17.1](https://github.com/sixohsix/twitter)

### Docker Option

Download [Docker](https://docs.docker.com/engine/installation/) and build the image as follows:
```
  docker build -t mining_the_social_web ~/Mining_The_Social_Web/
 ``` 

then, run the container with the instruction:
```
docker run --rm -it -v $(pwd):/Mining_The_Social_Web mining_the_social_web
```

and it should be ready to use.
