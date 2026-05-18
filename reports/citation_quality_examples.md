# Citation Quality Evaluation

This document covers milestone 5: evaluate 5 example RAG answers and check whether their citations support the generated response.

## Evaluation Setup

- Source notebook: `notebooks/Judy_RAG_Integration.ipynb`
- Answer output: `results/rag_sample_answers.csv`
- Retrieved evidence output: `results/rag_retrieved_contexts.csv`
- Evaluation questions: `evaluation/citation_quality_questions.json`
- Retriever: hybrid BM25 + MiniLM with Reciprocal Rank Fusion
- Generator used for this review: deterministic citation-grounded generator from `src/rag_pipeline.py`

Because the current StackLite corpus contains Stack Exchange question posts, the citation review focuses on whether the retrieved posts are relevant grounding evidence. The safest generated answer should avoid unsupported details when the retrieved snippet does not contain enough factual explanation.

## Rating Rubric

| Rating | Meaning |
|---|---|
| Good | The answer includes citations, the primary citation is directly relevant, and the answer does not add unsupported claims. |
| Needs improvement | The main citation is useful, but one or more supporting citations are broad, only partially relevant, or the answer is not direct enough. |
| Bad | The answer cites irrelevant documents, omits citations, or makes factual claims not supported by the cited evidence. |

## Summary

| # | Question | Top cited documents | Judgment |
|---:|---|---|---|
| 1 | Why are micro average precision recall and F1 equal in multiclass classification? | `datascience:15989`, `datascience:36817`, `datascience:40900` | Good |
| 2 | What is the difference between artificial intelligence and machine learning? | `ai:35`, `datascience:19077`, `ai:1847` | Good |
| 3 | What are deconvolutional layers in convolutional neural networks? | `datascience:6107`, `datascience:15903`, `datascience:51470` | Needs improvement |
| 4 | How do I set class weights for imbalanced classes in Keras? | `datascience:13490`, `datascience:44883`, `datascience:48369` | Good |
| 5 | What is the dying ReLU problem in neural networks? | `datascience:5706`, `datascience:18810`, `ai:40576` | Good |

## Detailed Examples

### 1. Micro Average Metrics

**Question:** Why are micro average precision recall and F1 equal in multiclass classification?

**Retrieved citations:**

- [1] `datascience:15989` - Micro Average vs Macro average Performance in a Multiclass classification setting
- [2] `datascience:36817` - Why is the F-measure preferred for classification tasks?
- [3] `datascience:40900` - Sklearn F1 score micro vs weighted for multiclass classification

**Judgment:** Good

**Reason:** The first citation is an exact match for the question, and the third citation is also about micro F1 in multiclass classification. The current generated answer is conservative and does not invent unsupported details.

**Good answer pattern:** Micro averaging computes global counts across classes before calculating precision, recall, and F1, so in a single-label multiclass setting those micro metrics can become equal; the exact StackLite discussion is the strongest citation [1], with micro-F1 context also supported by [3].

**Bad answer pattern:** Macro and micro averages are always identical for all classification problems [1].

**Why the bad version fails:** It overgeneralizes. The cited material is about a specific multiclass micro-average setting, not all averaging methods or all tasks.

### 2. AI vs Machine Learning

**Question:** What is the difference between artificial intelligence and machine learning?

**Retrieved citations:**

- [1] `ai:35` - What is the difference between artificial intelligence and machine learning?
- [2] `datascience:19077` - Difference between machine learning and artificial intelligence
- [3] `ai:1847` - Difference between artificial intelligence and cognitive science

**Judgment:** Good

**Reason:** The first two citations directly match the user question. The third citation is related to AI terminology but is weaker support for the AI-vs-ML distinction.

**Good answer pattern:** Artificial intelligence is the broader field of building systems that display intelligent behavior, while machine learning is a major approach within AI that learns patterns from data [1][2].

**Bad answer pattern:** AI and machine learning are exactly the same thing [1].

**Why the bad version fails:** The relevant citations discuss a distinction between the two terms, so saying they are exactly the same is unsupported.

### 3. Deconvolutional Layers

**Question:** What are deconvolutional layers in convolutional neural networks?

**Retrieved citations:**

- [1] `datascience:6107` - What are deconvolutional layers?
- [2] `datascience:15903` - Why do convolutional neural networks work?
- [3] `datascience:51470` - Differences between Convolutional1D, Convolutional2D, and Convolutional3D

**Judgment:** Needs improvement

**Reason:** The first citation is excellent because it exactly matches the question. The second and third citations are broader CNN-related posts and are not strong evidence for deconvolutional layers specifically.

**Good answer pattern:** Deconvolutional layers are discussed in the retrieved StackLite post about deconvolutional layers, which should be the primary citation [1]. Broader CNN citations should only be used as background if the answer explicitly needs them [2][3].

**Bad answer pattern:** Deconvolutional layers are fully explained by the difference between 1D, 2D, and 3D convolutions [3].

**Why the bad version fails:** Citation [3] is about convolution dimensionality, not the specific deconvolution-layer concept asked by the user.

### 4. Keras Class Weights

**Question:** How do I set class weights for imbalanced classes in Keras?

**Retrieved citations:**

- [1] `datascience:13490` - How to set class weights for imbalanced classes in Keras?
- [2] `datascience:44883` - Deep network not able to learn imbalanced data beyond the dominant class
- [3] `datascience:48369` - What loss function to use for imbalanced classes using PyTorch?

**Judgment:** Good

**Reason:** The top citation is an exact Keras class-weight match. The second and third citations support the general imbalanced-class context, but [3] is PyTorch-specific and should not be treated as Keras implementation evidence.

**Good answer pattern:** For Keras, the direct class-weight post should be the main source [1]. Related imbalanced-learning posts can help explain why class weighting matters, but they should be secondary citations [2][3].

**Bad answer pattern:** In Keras, you should use the PyTorch loss-weighting API to set class weights [3].

**Why the bad version fails:** The cited PyTorch post is not valid evidence for a Keras-specific implementation instruction.

### 5. Dying ReLU

**Question:** What is the dying ReLU problem in neural networks?

**Retrieved citations:**

- [1] `datascience:5706` - What is the "dying ReLU" problem in neural networks?
- [2] `datascience:18810` - How to check for dead relu neurons
- [3] `ai:40576` - Why use ReLU over Leaky ReLU?

**Judgment:** Good

**Reason:** The first citation directly matches the user question, and the second citation is strongly related because it discusses dead ReLU neurons. The third citation is relevant background about ReLU variants but should not be the primary evidence.

**Good answer pattern:** The dying ReLU problem refers to ReLU units becoming inactive or unhelpful during training; the direct dying-ReLU post is the primary evidence [1], and the dead-neuron diagnostic post is useful supporting evidence [2].

**Bad answer pattern:** Leaky ReLU is always better than ReLU and should replace ReLU in every model [3].

**Why the bad version fails:** It makes a broad recommendation that is not supported by the retrieved citation and goes beyond the user's question.

## Overall Findings

- All five examples include citation markers and source links.
- The top-ranked citation is directly relevant for all five questions.
- The current RAG answer style is safe because it avoids unsupported factual claims when snippets are limited.
- Citation quality is strongest when the top result exactly matches the question.
- Secondary citations sometimes become broad background evidence rather than direct support.

## Recommended Improvements

- Use only the top one or two citations when the third citation is broad or only loosely related.
- Add answer-aware citation filtering so each cited source supports a specific sentence.
- If accepted-answer text becomes available in the corpus, include it in the RAG context to generate more direct factual answers.
- Keep conservative generation when the retrieved snippets contain only question text and not enough explanatory answer content.
