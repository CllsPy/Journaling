# 🌟 **Study Report**

## Day 1: July 19, 2024

- **Large Language Models (LLMs)** are trained on vast amounts of data sourced from the internet, which means there is a high possibility of containing bias or inappropriate content.
- I learned that a software engineer must possess two principles: **self-learning** and **problem-solving**.
- A good software engineer is capable of delivering a **secure** and **functional** product, achievable through **Unit Testing**.
- How to make a Unit Test using Python:
  - 🐍 **Unit Testing in Python**
    - Use `python -m unittest test_module.py`
    - Common assertions: `assertEqual`, `assertRaises`
- I learned about **TDD (Test-Driven Development)**: A method to write software where you first write the tests, then write the code.
- **DRY (Don't Repeat Yourself)**
  - Your code should have as little repetition as possible.
  - Less repetition means fewer bugs.
- How to create a **function**.
- How to call a **function**.
- How to use **arrays** and operate them.
- **Unit Testing**
  - Use `python -m unittest test_module.py`
  - Common assertions: `assertEqual`, `assertRaises`.

## Day 2: July 20, 2024

- Learned how to authenticate HuggingFace using the login screen in Google Colab.
- Discovered the model [HuggingFaceTB/SmolLM-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM-1.7B-Instruct?text=write+a+poem).
- Explored the functionality of Prompt Template in LangChain.
- Understood the importance of standardizing prompts.
- Learned how to create an API and use it in Colab.
- Learned how the Prompt Template in LangChain works.
- Understood why it is important to standardize the prompt.
- Learned how to create an API and use it in Colab.

## Day 3: July 22, 2024
- How to create a Personal Access Token on GitHub
- How to use the PyGithub library
- How to mine GitHub using their API


**Machine Learning**

- How to train a Perceptron
- Bias, Batch Size
- Loss Function
- Activation Function

**matplotlib and numpy**
- np.meshgirid(x, y)
- plt.contourf()
- plt.clf()

**Resources**

- [REST API endpoints for users](https://docs.github.com/en/rest/users?apiVersion=2022-11-28)
- [Mining the social web](https://github.com/mikhailklassen/Mining-the-Social-Web-3rd-Edition)
- [Managing your personal access tokens](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)
- [Neural Networks and Perceptron - Lesson 9](https://www.youtube.com/watch?v=fEukSrpDPH0)
- [Perceptron For Dummies](https://jilp.org/cbp/Daniel-slides.PDF)
- [Perceptrons, Marvin](https://www.amazon.fr/Perceptrons-Intro-Computational-Geometry-Exp/dp/0262631113)
- [What is the purpose of meshgrid in NumPy?](https://stackoverflow.com/questions/36013063/what-is-the-purpose-of-meshgrid-in-numpy)

## Day 4: July 23, 2024

- I learned that there are five stages to training a Machine Learning model.
- I learned about the `bincount` command in Numpy, which counts the occurrence of similar values in an array.

**Sklearn**

- I learned how to implement the Perceptron algorithm in scikit-learn.
- I learned about the OvO and OvR techniques to enable models limited to binary classifications (Logit, SVM) to perform classification on multi-class data.
  
**Resources**

- [One-vs-Rest and One-vs-One for Multi-Class Classification](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)
- Page 503, Machine Learning: A Probabilistic Perspective, 2012.
- [Choosing the right estimator - Scikit-learn](https://scikit-learn.org/1.3/tutorial/machine_learning_map/index.html)

## Day 5: July 27, 2024

- I learned what Logistic Regression is.
- I learned what the Sigmoid Function is.
- I learned the difference between Gradient Descent and Stochastic Gradient Descent.

**Resources**
- [Tutorial 35 - Logistic Regression In-depth Intuition - Part 1 | Data Science](https://www.youtube.com/watch?v=L_xBe7MbPwk)
- [Tutorial 36 - Logistic Regression In-depth Intuition - Part 2 | Data Science](https://www.youtube.com/watch?v=uFfsSgQgerw)
- [Why Is Logistic Regression Called “Logistic Regression” And Not Logistic Classification?](https://medium.com/@praveenraj.gowd/why-is-logistic-regression-called-logistic-regression-and-not-a-logistic-classification-5a418293040d#:~:text=Linear%20regression%20gives%20a%20continuous,%E2%80%9CRegression%E2%80%9D%20in%20its%20name.)

## Day 6: July 31, 2024

- I learned about the concept of "maximum margin" in machine learning models.
- I understood that larger margins help reduce generalization error, while smaller margins increase the risk of overfitting or underfitting.
- I became familiar with the concept of "slack variables," introduced by W. Vapnik in 1995, which make SVM models more flexible when data is not linearly separable.
- I learned that these variables are represented by the parameter \( C \), which controls error tolerance and the trade-off between bias and variance.
- I realized that adjusting the value of \( C \) is crucial for ensuring the model generalizes well to new data, avoiding overfitting and underfitting.

**Resources**
- [StatQuest: Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8&list=PLblh5JKOoLUKxzEP5HA2d-Li7IJkHfXSe)
- Chris J.C. Burges’ excellent explanation in A Tutorial on Support Vector Machines for Pattern Recognition (Data Mining and Knowledge Discovery, 2(2): 121-167, 1998)
- Vladimir Vapnik’s book The Nature of Statistical Learning Theory, Springer Science+Business  Media, Vladimir Vapnik, 2000
- Andrew Ng’s very detailed lecture notes available at https://see.stanford.edu/materials/aimlcs229/cs229-notes3.pdf

## Day 7: Aug 1, 2024

- I learned that Logistic Regression tries to maximize conditional likelihoods.
- I became familiar with the concept of conditional likelihoods.
- I learned that Logistic Regression is more vulnerable to outliers than SVM.
- I understood that Logistic Regression is simpler to explain and therefore more attractive for streaming data.
- I learned about the SGDClassifier in scikit-learn, which allows for incremental training.
- I became familiar with the possibility of initializing the SGDClassifier with Perceptron, Logistic Regression, and SVM.
- I understood that we can use SVM with a kernel to solve nonlinear problems.
- I learned about kernel methods for linearly inseparable data.
- I comprehended the use of the kernel trick to classify nonlinear data in a high-dimensional space.
- I learned that the Gaussian Kernel is one of the most common kernel methods.

**Resources**
- [Likelihood Function](https://en.wikipedia.org/wiki/Likelihood_function)
- [Maximum Likelihood Estimation in Logistic Regression](https://arunaddagatla.medium.com/maximum-likelihood-estimation-in-logistic-regression-f86ff1627b67)
- [Machine Learning Notation](https://nthu-datalab.github.io/ml/slides/Notation.pdf)

## Day 8: Aug 3, 2024

- Today I learned about decision tree learning.
- Tree-based models are a good choice when interpretability is crucial.
- Data is classified based on the questions the model asks.
- The process starts at the top of the tree, selecting the feature with the highest Information Gain (IG).
- This process is repeated until the branches are pure, meaning a class has been chosen.
- In practice, this process can easily lead to overfitting.
- The goal is to choose the feature that provides the highest possible gain in information.
- To achieve this, an objective function is defined and optimized using the Decision Tree algorithm.
- Mathematically, Information Gain is the difference between the impurity of the parent node and the sum of the impurities of the child nodes.
- The lower the impurity of the children, the higher the gain.
- Computationally, it can be impractical, so sklearn uses the binary decision tree algorithm.
- The impurity measures used by binary decision trees include: Gini impurity (IG), Entropy (IH), and Classification Error (IE).

**Resources**
- [Binary Search](https://www.kaggle.com/discussions/accomplishments/523939#:~:text=Futher%20Readings-,Binary%20Search,-Add%20Tags)

## Day 9: Aug 4, 2024

- Random Forest is a technique that combines multiple decision trees.
- This model is considered an **ensemble** of decision trees.
- Random Forest averages the decision trees to generate a consistent result.
- The method helps to avoid overfitting.
- The technique consists of four main steps:
  1. Create a test set.
  2. Branch a decision tree for each test set and split the features at each node with the highest gain, based on the chosen metric.
  3. Repeat steps 1 and 2.
  4. Aggregate the results and choose the winning class by majority vote.
 

### papers
- [Generative AI Text Classification using Ensemble LLM Approaches](https://arxiv.org/pdf/2309.07755)

### Userful
- [Bloom](https://bigscience.huggingface.co/blog/bloom)
- [DetectGPT](https://detectgpt.com/app/)
- [Bootstrapping (statistics)](https://en.wikipedia.org/wiki/Bootstrapping_(statistics))


## Day 10: Aug 6, 2024

- I learned that Random Forest is an ensemble learning technique that combines multiple decision trees to generate more consistent results and avoid overfitting.
- I learned that the Random Forest process involves creating test sets, building a decision tree for each test set, repeating these steps, and finally aggregating the results through majority voting.
- I learned that reducing the sample size increases diversity and reduces the chances of overfitting, although it may negatively impact overall performance.
- I learned that K-nearest neighbors (KNN) is a lazy-learning algorithm that memorizes training data instead of optimizing a function.
- I learned that KNN is a non-parametric and instance-based algorithm, which adapts easily to new data but requires more resources as the amount of data increases.
- I learned that KNN classifies data by identifying the nearest elements and choosing the class based on majority voting.


### papers

- [Discriminant Function Analysis](https://www.sciencedirect.com/topics/neuroscience/discriminant-function-analysis#:~:text=Discriminant%20function%20analysis%20(DFA)%20is,normally%20distributed%20for%20the%20trait.)

### Userful

- [Bloom](https://bigscience.huggingface.co/blog/bloom)
- [DetectGPT](https://detectgpt.com/app/)
- [Bootstrapping (statistics)](https://en.wikipedia.org/wiki/Bootstrapping_(statistics))

## Day 11: Aug 7, 2024

- In the real world, data can be presented in different ways, generally irregular, missing values ​​are one of the manifestations, they can occur for several reasons; forgetfulness, non-existence of information; typing errors. There are techniques to deal with the absence of this data, such as deleting or replacing the data.
- We can use `isnull().sum()` to verify all null values on a dataframe
- To remove null data from our dataframe we can you the method .drop(), the method has some parameters i.e 'all' that removes only rows where all values are null.

## Day 12: Aug 9, 2024

- **I learned** that deleting missing numeric values can lead to significant information loss.
- **I learned** about interpolation, a common technique for filling missing data by replacing null values with the column’s mean.
- **I learned** that scikit-learn provides tools to perform interpolation of missing values.
- **I learned** how to use the scikit-learn API:
  - `data.fit`: Used to learn the parameters from the data.
  - `data.transform`: Applies the learned transformation.
  - `data.predict`: A method used by estimators to make predictions on test data.
- **I learned** that data consistency is important, as the array used in `fit` must have the same number of features when it is transformed.
- Today I learned that nominal classes do not need to follow any specific order and can start counting from 0.
- Today I learned that assigning ordered numerical values to nominal categorical features can lead to incorrect assumptions in machine learning models.
- Today I learned that one-hot encoding is a technique to prevent models from assuming an ordinal relationship among nominal categories by creating a binary representation for each category.

## Day 13: Aug 12, 2024

- Today I learned the importance of partitioning a dataset into separate training and test datasets, which allows us to evaluate the model's performance before deploying it in the real world.
- I learned about the benefits of bringing features onto the same scale, as most machine learning models perform better when the data is scaled.
- I learned that normalization typically involves limiting the range of values, usually between [0, 1], and can be achieved using the `MinMaxScaler` in scikit-learn.
- I also learned that scaling adjusts the proportion of the data, but it does not normalize the data.

## Day 14: Aug 19, 2024

A **Regularização** é um método que pune **weights** grandes, desse modo, a complexidade do modelo é diminuída; evitando assim, overfitting.

![](https://ars.els-cdn.com/content/image/3-s2.0-B978012823504100012X-f02-13-9780128235041.gif)


### References
- [Student](https://bayes.wustl.edu/Manual/Student.pdf)

## Day 15: Aug 20, 2024

**Sklearn** allows access to the values for **bias** and **weight** used by the model:

- intercept_ (bias)
- coef_ (weight)

The value of C and Regularization are closely connected in an inverse relationship: the smaller the value of C, the stronger the Regularization.

### Sequential feature selection algorithms

We can select features in two ways: extraction or selection. Selection involves choosing a few of the most relevant features from the dataset. On the other hand, extraction implies creating a new subset of features from the dataset.

When the function returns `self`, meaning the object itself, it indicates that we can apply a "chain of methods":
`obj.action1.action2...`.

## Day 16: Aug 23, 2024

- I read the paper [Attention is All You Need](https://arxiv.org/abs/1706.03762) and learned that:
  - The Transformer architecture achieves better results in translation tasks than RNNs.
  - Transformers consume fewer resources to train compared to other sequential models.

![](https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/The-Transformer-model-architecture.png/640px-The-Transformer-model-architecture.png)

- **[Hugging Face](https://huggingface.co/)** exists to solve the following dilemma: Training models from scratch or finding them in a framework different from the one you use can make the process stressful; Hugging Face simplifies this.

## Day 17: Aug 27, 2024

- **The encoder-decoder** architecture 
- The **Attention** mechanism
  
### References
- [Encoder-Decoder Architecture: Overview](https://www.youtube.com/watch?v=671xny8iows)
- [Encoder models](https://huggingface.co/learn/nlp-course/en/chapter1/5?fw=pt)
- [Transformer models and BERT model: Overview](https://www.youtube.com/watch?v=t45S_MwAcOw)
- [Attention Mechanism In a nutshell](https://www.youtube.com/watch?v=oMeIDqRguLY)

## Day 18: Aug 29, 2024

- The **Self-Attention** mechanism
- Hugging Face Pipelines
  - Sentiment Analysis
  - Summarization
  - Text Generation
  - NER (Named Entity Recognizion)
  - Translation
  - Q&A

- Create a water maker on a Jupyter Notebook:

```python
%load_ext watermark
%watermark -a "author" -u -d -p numpy, pandas
```

## Day 19: Set 02, 2024

- **`!wget`**: used to download files from the internet.
- **HuggingFace Pipeline:**
  - **Datasets**: load data
  - **Tokenizer**: tokenize data
  - **Transformers**: train/infer models
  - **Datasets**: evaluate

## Day 20: Set 03, 2024

- **map()** function
- AutoTokenizer from Hugging Face
- How load a dataset from Hugging Face Hub
- Tokenization
  - Word Tokenization
  - Character Tokenization
  - WordPiece Tokenization

## Day 21: Set 10, 2024

Today I Learnet about `non-linear` transformations and how we can take `no-linear`data, linear:
(xi, xii) -> (xi², xii²)

I use Chatgpt for build an webapp to create today datasets, I like this ideia and pretend to improve, create feature to select models and plot decision boudaries.


## Day 22: Sept 21, 2024

- The difference between `np.array([1, 2], (3, 4)]` and `np.array([[1,2], [3,4]])` is that in the first case, it raises and error, because np.array expect a single interable element (tuple, list).
- x.reshape(-1), reorganize the array in such way that it preservs its dimenions. Let's say, if we have an array (3,2) we end with and array (6,)
- (2, 3) = (rows, columns)
- (100, ) -> vector (1D)
- (100, 1) -> column vector (2D)

[Reshape -1, 1 and Reshape 1, -1 in Python NumPy | Module NumPy Tutorial - Part 07](https://www.youtube.com/watch?v=yDXNPyxDb0M)

## Day 23: Sept 22, 2024

- I'm reading [Handwritten Digit Recognition with a
Back-Propagation Network](https://proceedings.neurips.cc/paper/1989/file/53c3bce66e43be4f209556518c2fcb54-Paper.pdf)

- Today I learned how to use the diffuse mode to solve problems and I solved a problem when I was trying to build a NN to [predict handwritting digits (MNIST dataset) using TF](https://www.kaggle.com/code/carloscll/on-understanding-keras-flatten).
- I learned about the Learning Problem from the book "ML from data".

  ### pandas
- pd.concat for merge data.

## Day 24: Sept 24, 2024
- ∂C/∂w is the equation behide **backpropag**, It means how much C (loss function) changes when we change w and b.

### References
- [How the backpropagation algorithm works](http://neuralnetworksanddeeplearning.com/chap2.html)
- [Handwritten Digit Recognition with a Back-Propagation Network ](https://proceedings.neurips.cc/paper/1989/file/53c3bce66e43be4f209556518c2fcb54-Paper.pdf)
- [Difference Between Partial and Total Derivative](https://www.youtube.com/watch?v=Kp7sSp5Kn7o)

## 25: Sept 25, 2024

- Today I build a prompt for summarize papers, so I learned some building blocks for that as:
  - sugest report informations that the model don't understand (instead of make assumptios)
  - List tasks

 - I learned the notation for represente the a (activiation), w (eight) and b (bias) for a specific layer in a NN.
   
   ![image](https://github.com/user-attachments/assets/a971657c-5c17-4dcc-b8e0-0296ffab2384)

- I'm reading a book abour CS and some fundamental concepts, today I learned about the Von Neuman Arch
  
![image](https://github.com/user-attachments/assets/9c7f9b57-136e-4441-9d9f-836eda339be6)

## 26: Sept 26, 2024

- I learned how to use [Optuna](https://optuna-org.translate.goog/?_x_tr_sl=en&_x_tr_tl=fr&_x_tr_hl=fr&_x_tr_pto=sc) to find optimal parameters in ML models
- Learned how to ensemble results from threee models and using k-fold
  - xgb
  - catboost
  - lgbm 

## 27: Sept 76, 2024

- `np.choice` (random selection)
- `np.sort` (organize)
- `np.setdiff1d` (choice the difference between the lists)
- OOB Sample concept (out-of-the-bag) for create a test set
- Bootstrap sample

## 28: Sept 28, 2024

- I learned how to draft a essay and how it's important to learn something new.
- I learned eight [new things](https://arxiv.org/abs/2304.00612) about llm
  1. LLMs predictably get more capable with increasing investment, even without targeted innovation.
  2. Many important LLM behaviors emerge unpredictably as a byproduct of increasing investment.
  3. LLMs often appear to learn and use representations of the outside world.
  4. There are no reliable techniques for steering the behavior of LLMs.
  5. Experts are not yet able to interpret the inner workings of LLMs.
  6. Human performance on a task isn't an upper bound on LLM performance.
  7. LLMs need not express the values of their creators nor the values encoded in web text.
  8. Brief interactions with LLMs are often misleading.

## 29: Sept 29, 24

- **paper/essay**
  - how to create a reading list
    - how to take notes

- **ml**
- ml paradigms
  - supervised learning
  - cross-validation is a method to check model performance on real  unseen data
    - it splits data in train set and val set, the average of val set is use to verify performance


**references**
  - [ml foundatitions](https://www.hlevkin.com/hlevkin/45MachineDeepLearning/ML/Foundations_of_Machine_Learning.pdf)
- [cross-validation wikipedia](https://en.wikipedia.org/wiki/Cross-validation_(statistics)#cite_note-2)


## 30: Sept 30, 24

- there is a difference between **learning** and **design**
  - **learning** use data,  for design not necessarily]

  - pandas
    - map features

```python
map_feature = {
  0: 'a'
  1: 'b'
}

df.loc[:, 'colum'] = df.column.map(map_feature)
```

- overffiting and uderfitting
![image](https://github.com/user-attachments/assets/277f936c-9b28-4918-ba85-e600fd5829df)

- train improve and val dicrease (overfitting)

## 31: Oct 01, 24

- we can use cross-val for handle overfitting
  - hol-out set is a type of cv
  - each data needs a cv method (isn't global)

references
- https://thinkingneuron.com/how-to-test-ml-models-using-k-fold-cross-validation-in-python/
- https://thinkingneuron.com/how-to-test-machine-learning-models-using-bootstrapping/


## 32: Oct 02, 24

- for select a inverval of cols witth pandas
    - df.iloc[:, 1:3]
    - df.loc[:, 'A', 'B']

 ## 33: Oct 03, 24

- -1 use all available CPUs when you using cross-val.

references
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html
- https://arxiv.org/pdf/1811.12808


## 34, Oct 04 -2024

- tensorflow
  - how to create a model using keras.Sequential
  - what's a layer: a filter for data (data go trought and get better)


## 35, Oct 05 - 2024
- tensor
  - a tensor is a multi-dimensional array
  - how to slice a tensor
  - [:] represents all the axis
  - a tensor has three thigs
    - ndim
    - shape
    - dtype
   

## 36, Oct 05 - 2024
- keras
- keras.layers receive a matrix and outputs a matrix
  - output = relu(dot(input, w) + b)

## 37, Oct  07 - 2024
- ensemble
  - bagging

## 38, Oct 08 - 2024
- npl with ml
  - bag of words
 
  - test system
      - quizz (replicate task on another dataset)
      - mini-essay
      - test: Kaggle 10%
      - essay about the topic

- c
  - hello world

### references
- [Training vs testing](https://www.youtube.com/watch?v=SEYAnnLazMU&list=PLD63A284B7615313A&index=5)
- [CS50](https://www.youtube.com/watch?v=LfaMVlDaQ24&t=7550s)

## 39, Oct 09 - 2024

- adversarial validation

## 40, Oct 11 - 2024

- bagging
  - advanced bagging
  - how to plot model predictions

- random forest

  ![image](https://github.com/user-attachments/assets/6a7218bb-b8ac-41b9-a0c7-c4f7d1d64a75)

**references**
  - [bagging sklearn](https://inria.github.io/scikit-learn-mooc/python_scripts/ensemble_bagging.html)


## 41, Oct 14, 2024
- how to use mlxtende for eda

**references**
- [mlxtende](https://rasbt.github.io/mlxtend/)

## Day 42: Oct 14 - 2024

- sklearn
  - method get_params() to basEestimator
 
- python
- dunder methods
    - __get_item__
    - __len__
- namedtuple
  - Student = namedtuple('Student', ['name', 'age', 'DOB'])

- cs
  - imperative  paradigms
  - declarative paradigms
**resources**
- [sklearn baseEstimator](https://scikit-learn.org/1.5/modules/generated/sklearn.base.BaseEstimator.html)


## Day 43: Oct 18 - 2024

- numpy
  - `random.norm` - create a normal gaussian distribution
  - `ndarray.flatten(order='C')` colapse the array to one dimenstion
 
- sklearn
- `inverse_transform`: remove transformation (e.g StandardScaler())

- references
  - https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
  - https://numpy.org/doc/2.0/reference/generated/numpy.ndarray.flatten.html
  - https://stackoverflow.com/questions/49885007/how-to-use-scikit-learn-inverse-transform-with-new-values


## Day 44: Oct 20 - 2024
- ransac
  - inliers
  - MAD (mean abs desviation)


- references
  - [MAD](https://www.khanacademy.org/math/statistics-probability/summarizing-quantitative-data/other-measures-of-spread/a/mean-absolute-deviation-mad-review)
  - [RANSAC](https://en.wikipedia.org/wiki/Random_sample_consensus)

## Day 45: Oct 21 - 2024

- numpy
  - nump.max
  - numpu.min

## Day 46: Oct 23, 2024

- Learned how to build a running sum with python, for a leetcode problem.

```python
class Solution:
    def runningSum(self, nums: List[int]) -> List[int]:
        result = [nums[0]]

        for i in range(1, len(nums), 1):
            result.append(nums[i] + result[i-1])
        
        return result
```

Links

- [Sum Of 1d Array](https://leetcode.com/problems/running-sum-of-1d-array/)

## Day 47: Oct 26, 2024

- Learned about Role Prompt, a Prompt Engineer technique
  - Normal Prompt: what do you thinkg about skate board?
  - Role Prompt: as I cat, what do you thing about skate board?

Links
- [Prompt Engineer](https://roadmap.sh/prompt-engineering)
- [Learn Prompting](https://learnprompting.org/blog/2024/9/29/the_prompt_report)

## Day 48: Oct 27, 2024
- Python is a scripting and programming language
  - You can use Python like Java, C++ and others pragramming languages, because of the OOP principle and Python constructors.

- regulaziation
  - We use Regularization when we need reduce complexity, in our case model complexity.
  - You can divide regularization in two categories:
    - explicity: adding new information to the loss function (as penalty)
    - implicity: tecnhiques that reduce model capacity to learn noise like a more robust loss function to outliers, model combination to select the most pluasible answer (ensemble).

- loss
  - The loss function is the mesure we use to update our wights and bias.
  - The loss function is used to measure the error in one pair (x, y) at time. The cost function on the other hand is used for calculate the overall error (based on all data points).


  ## Day 49: Oct 28, 2024
  
- l2 regularization
  ![image](https://github.com/user-attachments/assets/1a0e599b-061a-4b61-8d49-f9aeeff2e4ba)

  - what's regularization
    - Is aditional information to adjust the loss.

- huggingface
  - to use hf models with the pipe api you need three elementos:
    - model
    - task
    - pipe(content)

```python
model = pipeline('summarization', model='bERT')
model('text-content')
```
  Links
- https://developers.google.com/machine-learning/crash-course/overfitting/regularization
- https://developers.google.com/machine-learning/crash-course/overfitting/model-complexity
- https://en.wikipedia.org/wiki/Occam's_razor
- https://developers.google.com/machine-learning/glossary#l2-regularization
- https://en.wikipedia.org/wiki/Regularization_(mathematics)
- https://www.datacamp.com/tutorial/loss-function-in-machine-learning
- https://stats.stackexchange.com/questions/359043/what-is-the-difference-between-a-loss-function-and-an-error-function
- https://developers.google.com/machine-learning/crash-course/linear-regression/gradient-descent
- https://huggingface.co/docs/transformers/main_classes/pipelines

  ## Day 50: Oct 31, 2024

  - learnt how to loop a nested list while i was doing leetcode
  - understood the bayes' theorm statement
  - discovered what is spaCY and nltk

  - huggingface
    - how to use ```text2image``` model
    - how to use langchain and huggingface
    - how to use ```text2text``` model

  Links
  - https://web.stanford.edu/class/cs224g/slides/Fulll%20Stack%20LLMs_%20Stanford%20University.pdf
  - https://fullstackdeeplearning.com/course/2022/lecture-1-course-vision-and-when-to-use-ml/
  - [3B1B - Bayes' Theorem](3blue1brown.com/lessons/bayes-theorem)
  - https://roadmap.sh/prompt-engineering

## Day 51: Nov 04, 2024

- practice with huggingface apis
  - tasks
  - pipeline
    - text
    - img2text
  - transformers
  - not confident about the audio task; fine-tune models.

## Day 52: Nov 05, 2024

- learnt how to use gpu for hf models

```python
device = "cuda:0" if torch.cuda.is_available() else "gpu"
model = pipeline("task", model="model_choice", device=device)
```

- how acess secret key on gcolabs

```python
from google.colabs import userdata
import os

sec_key = userdata.get("HF_TOKEN")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = sec_key
```

## Day 53: Nov 07, 2024

- learned about sorted funciton in python
  - use to organize lists
  - sorted(value, key, reverse)
  - you need a lista, key and reverse are optional (key organize based on a personal funcion and reverse (invert the order))

Links
- https://www.w3schools.com/python/module_statistics.asp

## Day 54: Nov 09, 2024
- learnt about typing hints: typing hits are just for documentation since python is a dimanic language
- Weaks
  - I had difficult to understand typing Lists.
  - zip()

```python
l1 = [1, 2, 3]
l2 = ['A', 'B', 'C']

for j, k in zip(l1, l2):
    print(f'zipped:{j, k}')

```

Links
- https://www.youtube.com/watch?v=QORvB-_mbZ0
- [How use typing for lists](https://youtu.be/QORvB-_mbZ0?si=ifsqaMHqwjeI3D5V&t=541)
  
## Day 54: Nov 13, 2024
- Build my first regression model with TF for get my FCC certification
- Understand the difference between NN and DNN
- Understand how to use in TF
  - Regularization
  - Early_Stop


Links
- https://www.tensorfalow.org/tutorials/keras/regression#linear_regression_with_multiple_inputs


## Day 55: Nov 16, 2024

python
- data structures
  - Lists and methods
  - list comprenhension
 
```python
[x+1 for x in [1, 2, 5]
```

  - Dict and methos
  - Set and methods
  - Collections
    - namedtuple

## Day 56: Nov 17, 2024

streamlit
- how to change theme
  - config.toml
- how to personalize your footer
- how to remove streamlit default footer
- how use gitpod.io to modify your code
- why is st.sessions and how to use

github
- recorded how to merge brachs
- recorded how to clone a repo

pylint
- how to use
- what is pylint


## Day 57: Nov 18, 2024

Google Style Guide

- import the entire module, not a specific action.
	- good: import scripts
	- bad: from scripts import abc


 Streamlit
   - How to use containers
   - What's a placeholders

Github
- How use code-spaces
- How to open and close a issue
- How to create a new branch

```bash
git switch -b <branch>
```

Links
- https://www.datacamp.com/tutorial/git-switch-branch

## Day 58: Nov 19, 2024

python

- pip install -U: U is a flag for upgrade
- Project Structure

```python

└── src/
    ├── main.py
    └── utils/
        ├── __init__.py
        └── util.py
```

- if __name__ == '__main__' is used to control the behave of a script when other scripts call it or when we run directly.
- If __name__ == '__main__' is userful when you need reuse the code e.g for testing.

Links
- https://iq-inc.com/importerror-attempted-relative-import/

## Day 59: Nov 20, 2024

python
- pip install -U: U is a flag for upgrade
- Project Structure

Links
- https://superlearninglab.substack.com/p/how-to-learn-anything-with-ultra
- https://iq-inc.com/importerror-attempted-relative-import/

## Day 60: Nov 21, 2024

**Machine Learning**
- There are three types of ML
	- Supervised: has labels in it
		 - Classification: predict a class (e.g rain/no-rain)
		 - Regression: predict numerical value
	- Unsupervised: not labels at all
	- Reiforcement: model learn by trying

- We apply ML when we want to find a g(x) that approximates f(x).


Links
- [Google for Devs](https://developers.google.com/machine-learning/intro-to-ml/what-is-ml)


## Day 61: Nov 22, 2024

RFL
- What's Q-Learning
	- How to apply Q-Learning
   
- Wht's Openai Gym
	- How to use Openai Gym on GColabs

  - [rfl datacamo q-learning](https://www.datacamp.com/tutorial/introduction-q-learning-beginner-tutorial)
  - [rfl datacamp: what's rfl](https://www.datacamp.com/tutorial/reinforcement-learning-python-introduction)


## Day 62: Nov 23, 2024

streamlit
- Learnt how to create pages and subpages

numpy
- np.argsort organize a array by it's indice, lets say

```python
>> array = [3, 2, 1]
>> np.argsort(array)
```
... [2, 1, 0]

- genetic algorithm: works a natural selection, you compare one solutions with another solutions and choose a winner.

Links

- https://numpy.org/doc/stable/reference/generated/numpy.argsort.html
- https://www.datacamp.com/tutorial/genetic-algorithm-python
- https://amberlemckee.weebly.com/data-projects--tutorials.html
- https://www.youtube.com/@DigitalSreeni


## Day 63: Nov 29, 2024

spacy
- how to apply basic text operations with spacy
	- POS Tagging
	- Tokenization
	- Lemmatization


## Day 64: Dez 01, 2024

nlu metrics
- [Glue](https://gluebenchmark.com/leaderboard/)
- [Superglue](super.gluebenchmark.coam)

python
- use if __name__ when you don't want execute from the module you are just importing.

ml
- conceps
	- decision function
	- net input
	- mcp
	- perceptron learning rule

Links

- https://www.alura.com.br/artigos/o-que-significa-if-name-main-no-python



## Day 64: Dez 03, 2024

**Golang**
- loops
- func
- var
- :=
- types (float32, float64, int, string)
- build two functions to operate vectors

**spacy**
- review
- pos_
- tagging
- lammanization
- container
- tokenization
- children
- parsing


Links

- [spacy course](https://course.spacy.io/pt/)


## Day 65: Dez 06, 2024

**new**

- start cs224n (stanford nlp_
- start cs50 with Python

**review**

- topic modeling
	- tf-idf
	- bi-gram; tri-gram
 
 - haddling-text
	 - re.sub()
	 - replace()

Links

- https://www.youtube.com/watch?v=rmVRLeJRkl4&list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4
- https://www.youtube.com/watch?v=LfaMVlDaQ24&t=7733s
- https://www.youtube.com/watch?v=3xaVX0cluDo&t=162s

Reading List

- [w2vec](https://arxiv.org/pdf/1301.3781)


## Day 66: Dez 07, 2024


**cs229 - machine learning**

- learnt about gradient and hessians
- solve a problem from problem set 0

Links

- [Vector and matrix derivatives](https://www.youtube.com/watch?v=FCWrduAxf-Q&t=159s)
