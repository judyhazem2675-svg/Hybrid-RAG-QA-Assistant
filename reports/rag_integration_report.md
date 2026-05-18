# RAG Integration Demo

This report covers Judy Hazem's RAG integration milestone.

## Method

- Corpus: `data/stacklite_questions.csv`
- Lexical retriever: Okapi BM25
- Dense retriever: `sentence-transformers/all-MiniLM-L6-v2`
- Fusion: Reciprocal Rank Fusion over BM25 and semantic results
- Generator: `extractive`
- Note: StackLite provides question-post passages, so citations point to retrieved posts used as grounding.

## Sample Answers

### Question 1

Why are micro average precision recall and F1 equal in multiclass classification?

The retrieved StackLite evidence points to the following cited source passages as the best grounding for the answer.

[1] Micro Average vs Macro average Performance in a Multiclass classification setting: Micro Average vs Macro average Performance in a Multiclass classification setting Micro Average vs Macro average Performance in a Multiclass classification setting I am trying out...
[2] Why is the F-measure preferred for classification tasks?: Why is the F-measure preferred for classification tasks? Why is the F-measure preferred for classification tasks? Why is the F-measure usually used for (supervised) classification...
[3] What's the difference between Sklearn F1 score 'micro' and 'weighted' for a multi class classification problem?: What's the difference between Sklearn F1 score 'micro' and 'weighted' for a multi class classification problem? What's the difference between Sklearn F1 score 'micro' and 'weighted...

For the question 'Why are micro average precision recall and F1 equal in multiclass classification?', use these citations as the supporting evidence and avoid adding claims that are not supported by the retrieved passages.

**Citations:** [1] datascience:15989, [2] datascience:36817, [3] datascience:40900

### Question 2

What is the difference between artificial intelligence and machine learning?

The retrieved StackLite evidence points to the following cited source passages as the best grounding for the answer.

[1] What is the difference between artificial intelligence and machine learning?: What is the difference between artificial intelligence and machine learning? What is the difference between artificial intelligence and machine learning? These two terms seem to be...
[2] Difference between machine learning and artificial intelligence: Difference between machine learning and artificial intelligence Difference between machine learning and artificial intelligence Is there any difference between machine learning and...
[3] What is the difference between artificial intelligence and cognitive science?: What is the difference between artificial intelligence and cognitive science? What is the difference between artificial intelligence and cognitive science? Sometimes I understand t...

For the question 'What is the difference between artificial intelligence and machine learning?', use these citations as the supporting evidence and avoid adding claims that are not supported by the retrieved passages.

**Citations:** [1] ai:35, [2] datascience:19077, [3] ai:1847

### Question 3

What are deconvolutional layers in convolutional neural networks?

The retrieved StackLite evidence points to the following cited source passages as the best grounding for the answer.

[1] What are deconvolutional layers?: What are deconvolutional layers? What are deconvolutional layers? I recently read Fully Convolutional Networks for Semantic Segmentation by Jonathan Long, Evan Shelhamer, Trevor Da...
[2] Why do convolutional neural networks work?: Why do convolutional neural networks work? Why do convolutional neural networks work? I have often heard people saying that why convolutional neural networks are still poorly under...
[3] What are the differences between Convolutional1D, Convolutional2D, and Convolutional3D?: What are the differences between Convolutional1D, Convolutional2D, and Convolutional3D? What are the differences between Convolutional1D, Convolutional2D, and Convolutional3D? I've...

For the question 'What are deconvolutional layers in convolutional neural networks?', use these citations as the supporting evidence and avoid adding claims that are not supported by the retrieved passages.

**Citations:** [1] datascience:6107, [2] datascience:15903, [3] datascience:51470

### Question 4

How do I set class weights for imbalanced classes in Keras?

The retrieved StackLite evidence points to the following cited source passages as the best grounding for the answer.

[1] How to set class weights for imbalanced classes in Keras?: How to set class weights for imbalanced classes in Keras? How to set class weights for imbalanced classes in Keras? I know that there is a possibility in Keras with the class_weigh...
[2] Deep network not able to learn imbalanced data beyond the dominant class: Deep network not able to learn imbalanced data beyond the dominant class Deep network not able to learn imbalanced data beyond the dominant class I have data with 5 output classes....
[3] What loss function to use for imbalanced classes (using PyTorch)?: What loss function to use for imbalanced classes (using PyTorch)? What loss function to use for imbalanced classes (using PyTorch)? I have a dataset with 3 classes with the followi...

For the question 'How do I set class weights for imbalanced classes in Keras?', use these citations as the supporting evidence and avoid adding claims that are not supported by the retrieved passages.

**Citations:** [1] datascience:13490, [2] datascience:44883, [3] datascience:48369

### Question 5

What is the dying ReLU problem in neural networks?

The retrieved StackLite evidence points to the following cited source passages as the best grounding for the answer.

[1] What is the "dying ReLU" problem in neural networks?: What is the "dying ReLU" problem in neural networks? What is the "dying ReLU" problem in neural networks? Referring to the Stanford course notes on Convolutional Neural Networks fo...
[2] How to check for dead relu neurons: How to check for dead relu neurons How to check for dead relu neurons Background: While fitting neural networks with relu activation, I found that sometimes the prediction becomes...
[3] Why use ReLU over Leaky ReLU?: Title: Why use ReLU over Leaky ReLU? Body: From my understanding a leaky ReLU attempts to address issues of vanishing gradients and nonzero-centeredness by keeping neurons that fir...

For the question 'What is the dying ReLU problem in neural networks?', use these citations as the supporting evidence and avoid adding claims that are not supported by the retrieved passages.

**Citations:** [1] datascience:5706, [2] datascience:18810, [3] ai:40576

## Output Files

- Answers: `results/rag_sample_answers.csv`
- Retrieved contexts: `results/rag_retrieved_contexts.csv`
