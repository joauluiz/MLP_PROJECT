# MACHINE LEARNING PROJECTS

During my graduation I had the opportunity to work with machine learning projects for 3 years. It was a challenging job learn a little about this universe of machine learning. I worked with 2 neural networks (MLP - Multilayer Perceptron and ELM - Extreme Learning Machine). I used them for prediction of hospitalization admission, using these variables (inputs): pollutant concentration (Particle Matter), avarege temperature, avarege umidity, day of the week and the classification if the day was a regular day or not. The output was just the number of hospital admission for the São Paulo city, the biggest city in Brazil. The data are from 01/01/2014 to 31/12/2016. 

# Implementation Informations

The IDE used for these projects was the Spyder (for better variables values visualizations) and also for multiples graphs plotted. The code language used was Python. The main libraries used were Pandas, Numpy, Sklearn and Matplotlib. For a better performance of the neural networks, it was made the k fold cross-validation process, which helps to discover what part of the dataset has a better generalization for the model. The metrics used to analyze the results were the MAE (Mean Absolute Error) and MSE (Mean Squared Error). Lower the values of these metrics, better are the results of the prediction. For this it study it was considered lags for the impact of the pollutant in the human health. For example, if a person has contact with a pollutant in a determinated day, they can show the side effects some days later. THE CODES STILL NEED SOME COMMENTS FOR A BETTER UNDERSTANDING FOR WHO'S READING, THIS IS SOMETHING I'LL DO SOON. I DECIDED TO PUT IN THE GITHUB ALREADY TO FINISH THE CREATION OF THIS REPOSITORY THAT I HAVE BEEN TAKING TOO LONG TO FINISH BECAUSE OF THE TERMINATION OF THE COLLEGE AND THE TERMINATION AS WELL OF MY INTERNSHIP.

# About the training, validation and test

The training set is used to train the model, which involves adjusting the model's parameters in order to minimize the error between the predicted output and the true output. The model is typically trained using an optimization algorithm, such as gradient descent, which adjusts the parameters to minimize the error.

The validation set is used to evaluate the model's performance during the training process. It is used to fine-tune the model's hyperparameters, which are parameters that are not learned directly from the training data, but rather set by the practitioner. For example, the learning rate and the regularization strength are common hyperparameters that can be adjusted using the validation set.

The test set is used to evaluate the final performance of the model, after it has been trained and the hyperparameters have been set. It provides an unbiased estimate of the model's performance on unseen data.

Overall, the training, validation, and test sets are used to evaluate the performance of a machine learning model at different stages of the development process. They help to ensure that the model is able to generalize well to new data, and can be used to identify any issues or overfitting during the training process.

# Multilayer Perceptron

A multilayer perceptron (MLP) is a type of artificial neural network that is composed of multiple layers of interconnected "neurons." It is called a multilayer perceptron because it is composed of at least three layers: an input layer, one or more hidden layers, and an output layer.

The input layer receives the input data, and each neuron in the hidden and output layers performs a simple mathematical operation on the data it receives from the previous layer, called "propagation." The output of each neuron is passed through an activation function, which determines the final output of the neuron. The final output of the MLP is produced by the output layer, which contains one or more neurons.

MLPs are used for a wide range of tasks, including classification, regression, and dimensionality reduction. They have been used extensively in image and speech recognition, natural language processing, and other areas of artificial intelligence.

# Extreme Learing Machine

An extreme learning machine (ELM) is a type of artificial neural network that is designed to be easy to train and fast to use. It is called an "extreme" learning machine because it is designed to learn from very large datasets quickly, without the need for traditional optimization algorithms like gradient descent.

One disadvantage of ELM is that it can be less accurate than other types of neural networks, particularly when the training data is noisy or there are a large number of features. However, it can be a good choice for applications where speed is a priority, such as real-time classification or prediction tasks.
