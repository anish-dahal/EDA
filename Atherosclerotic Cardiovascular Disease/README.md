# Machine Learning Algorithm from scratch

### Introduction

This project was done to reflect the total understanding of python for machine learning implementation. In this project, we use almost all of the techniques that were learned from the previous task. For this project selected dataset is HeartStudy.csv which is used for multiclass classification problems.
The dataset contains various features which can be used to predict the chance of people suffering from Atherosclerotic Cardiovascular Diseases (ASCVD). The main task is to implement one of the classification algorithms from scratch and use at least 3 classification algorithms provides by sklearn libraries and view the evaluation metrics.

### Implementation

First, for a better understanding of the provided dataset, the EDA technique was used. From this, we learn what type of encoding process should be used, as well as the presence of outliers and missing values in the dataset. After getting the information about the dataset we use three machine learning algorithms and see which one provides the best answer for the imbalance dataset. We also implement a logistic regression algorithm for the multiclass problem using the softmax activation function.

**Multiclass logistic regression workflow**

Let `X,` be the input matrix with `m` number of features and `W` be the trainable parameters and `Y` be the output label of `n` classes. Then Step

1. Initialize parameter `W` randomly as with shape `(m, n)`.
2. Calculate the product of `X` and `W`. `Z = X.W`
3. Take the softmax for each row *Zi*
$P_i = softmax(Z_i) = {\exp(Z_i) \above{1pt} \sum_{k=o}^{n}{\exp(Z_{ik})}}$
4. Update parameter `W`by using gradient decent algorithm using any loss function `L`
$W := W - \alpha {\partial L \above {1pt}\partial W}$
Here $\alpha$ is the learning rate
5. Iterate step 2,3, and 4 multiple times to make the loss function `L`as minimum as possible.
6. Select the index of highest probability from probability $P_i$ and the highest probability index is the predicated class index.

### Takeaways

- A better understanding of EDA.
- Use of correct metrics for classification problems.
- Write code in pythonic ways. Use of docstring, comments, and implement the class method.
- Understanding of classification algorithm and selecting of algorithm based on the dataset.