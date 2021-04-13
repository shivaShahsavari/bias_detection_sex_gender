# Bias detection against sex and gender  
Finding disposition against sex and gender in text has been a recent research concept in Natural language processing (NLP) area. This concept was started by social media such as Twitter and Facebook at 2015. These popular companies actively kicked off seeking the hate speech among their users and announced that they would seek to combat racism and xenophobia aimed at refugees at 2016.  
Considering this background, two reseachers named Zeerak Waseem and Dirk Hovy from Copenhagen University of Denmark provided a dataset of 16k tweets annotated for hate speech. This dataset was used later by other reseachers in NLP area to optimize their offensive recognition/bias detection models. To download this dataset please refer to this link : https://github.com/ZeerakW/hatespeech  
This dataset only includes twitter Ids and the sexism/racism labels. In order to gather twitter text, you need to get a twitter developer account. To fetch data by Ids easier, you can use **Twitter API version2** developed in this link: https://github.com/twitterdev/Twitter-API-v2-sample-code  
One of a few researches in this concept was done in Standford University at 2018. The researchers applied CNN with different Kernel sizes and could reach higher F1 score in comparison to the previous works. their CNN architecture is shown in below image. [S Paul, J Bhaskaran (2018), ERASeD: Exposing Racism and Sexism using Deep Learning]

<imge src="image/CNN_architecture_topic_classification.png" height=300 width=500>

In this project, we also implemented the same **CNN architecture with kernel size=2,3,4,5**  
Python libraries used in this project : **Numpy, Pands, sklearn, NLTK, Keras, Tensorflow**  
In each 10 traning epochs, separate precisions and recalls are calculated by callback and at the end, F1 measure is calculated by mean of all precisions and recalls.
