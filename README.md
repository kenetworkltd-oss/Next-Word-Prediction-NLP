# Next-Word Prediction Language Model using PyTorch

A comparative Natural Language Processing (NLP) project that explores next-word prediction using four deep learning architectures: Simple RNN, LSTM, GRU, and Transformer.

The project uses the WikiText-2 dataset and PyTorch to investigate how recurrent and attention-based neural networks learn sequential language patterns.

## Project Overview

Next-word prediction is a language modelling task in which a model receives a sequence of preceding tokens and attempts to predict the next token.

For example:

**Input:** `the united states`

**Predicted next word:** `rifles`

The objective of this project was to understand the fundamentals of language modelling, implement different neural network architectures, compare their performance, and experiment with text generation.

This is an educational language modelling project. It is not a large language model (LLM) or a production-ready text generation system.

## Objectives

- Understand NLP text preprocessing and tokenisation.
- Convert text into numerical representations suitable for neural networks.
- Build and train recurrent neural networks using PyTorch.
- Understand the differences between RNN, LSTM, GRU, and Transformer architectures.
- Implement positional embeddings and causal attention masking.
- Compare model performance using training and validation metrics.
- Apply early stopping and experiment with hyperparameter tuning.
- Evaluate the selected model on an unseen test dataset.
- Implement next-word prediction and basic text generation.

## Dataset

**Dataset:** WikiText-2

**Source:** Hugging Face Datasets — `Salesforce/wikitext`

**Configuration:** `wikitext-2-v1`

The official training, validation, and test splits were used.

### Dataset preparation

The preprocessing workflow included:

1. Loading the official dataset splits.
2. Inspecting the dataset and its text content.
3. Removing empty text rows.
4. Converting text to lowercase.
5. Preserving punctuation and WikiText special tokens.
6. Building the vocabulary using training data only.
7. Converting tokens into numerical IDs.
8. Creating fixed-length input sequences and next-token targets.

### Dataset statistics

| Dataset component | Number of rows |
|---|---:|
| Training | 23,767 |
| Validation | 2,461 |
| Test | 2,891 |

| Preparation detail | Value |
|---|---:|
| Vocabulary size | 28,912 |
| Context length | 30 tokens |
| Training sequences | 1,531,747 |
| Validation sequences | 158,707 |
| Test sequences | 178,727 |

The vocabulary includes the `<PAD>` token. The `<unk>` token is used for words outside the vocabulary.

The test split was held out during model training and validation and used for final evaluation.

## Model Architectures

Four neural network architectures were implemented and compared.

### 1. Simple RNN

A basic recurrent neural network that processes sequential information using hidden states.

### 2. LSTM

Long Short-Term Memory networks use input, forget, and output gates to manage information across time steps and help address the vanishing gradient problem.

### 3. GRU

Gated Recurrent Units use update and reset gates to control information flow through the network.

### 4. Transformer

An attention-based architecture using:

- Token embeddings.
- Positional embeddings.
- Multi-head self-attention.
- Transformer encoder layers.
- Causal attention masking.
- A linear output layer for next-token prediction.

The Transformer was included to compare attention-based sequence modelling with recurrent architectures.

## Training and Evaluation

The models were implemented using PyTorch.

The training workflow included:

- Training and validation loops.
- Cross-entropy loss.
- Adam optimisation.
- Training and validation accuracy tracking.
- Early stopping based on validation loss.
- Model checkpoint saving.
- Final model selection using validation performance.

The same dataset preparation and next-token evaluation approach were used across the architectures.

## Model Comparison

The following results were recorded at each model's best validation-loss epoch.

| Model | Best Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
|---|---:|---:|---:|---:|---:|
| Simple RNN | 20 | 5.1749 | 20.52% | 5.7882 | 18.12% |
| LSTM | 4 | 5.0752 | 23.19% | 5.4009 | 21.46% |
| GRU | 3 | 5.0640 | 22.50% | 5.3351 | 21.29% |
| Transformer | 6 | 5.0424 | 21.94% | 5.2736 | 21.18% |

### Comparison observations

- The Transformer achieved the lowest validation loss of 5.2736.
- The LSTM achieved the highest validation accuracy at its best validation-loss checkpoint.
- The Simple RNN performed less effectively than the other three architectures.
- The Transformer was selected for further experimentation because of its validation loss.

## Hyperparameter Tuning

A tuning experiment was performed on the baseline Transformer.

The dropout rate was increased from 0.1 to 0.2 while the learning rate remained at 0.001. Other architecture settings were retained.

| Transformer configuration | Best validation loss | Best epoch |
|---|---:|---:|
| Baseline Transformer | 5.2736 | 6 |
| Tuned Transformer | 5.3086 | 2 |

The tuning experiment did not improve validation loss.

The original baseline Transformer was therefore retained as the final model.

## Final Test Evaluation

The selected baseline Transformer was evaluated on the held-out test dataset.

### Final test results

| Evaluation metric | Result |
|---|---:|
| Test Loss | 5.2073 |
| Test Accuracy | 21.55% |
| Test Perplexity | 182.60 |
| Correct Predictions | 38,509 |
| Total Predictions | 178,727 |
| Weighted Precision | 15.63% |
| Weighted Recall | 21.55% |
| Weighted F1-Score | 15.61% |

### Interpretation

The final model correctly predicted the exact next token in approximately 21.55% of the test examples.

The test perplexity of 182.60 indicates substantial uncertainty when predicting the next token.

The weighted precision and F1-score were also relatively low, indicating that prediction performance was limited across the vocabulary.

These results show that the model learned some sequential language patterns, but its next-token prediction performance remains modest.

The results should be interpreted as an educational baseline rather than evidence of a highly capable language model.

## Next-Word Prediction and Text Generation

The selected Transformer checkpoint was loaded for inference.

A reusable prediction function was implemented to:

1. Accept a text prompt.
2. Tokenise the input.
3. Convert tokens into vocabulary IDs.
4. Retain the most recent 30 tokens.
5. Generate embeddings and positional information.
6. Apply causal masking.
7. Obtain the model's next-token prediction.

Both greedy prediction and temperature-based sampling were explored.

### Observed limitations during generation

Generated text sometimes contained:

- Repeated words and phrases.
- Weak sentence structure.
- Inconsistent contextual relevance.
- Unknown tokens represented by `<unk>`.

For example, a generated sequence beginning with `the united states` repeatedly produced words such as `rifles` and `regiment`.

Temperature sampling introduced some variation but did not consistently produce coherent text.

## Limitations and Challenges Encountered

This project provided practical experience, but several limitations affected the final results.

### 1. Limited predictive performance

The final Transformer achieved 21.55% test accuracy and a perplexity of 182.60.

The model was not sufficiently accurate or coherent for reliable general-purpose text generation.

### 2. Vocabulary size

The vocabulary contained 28,912 tokens.

Predicting among a large number of possible tokens makes exact next-token prediction challenging, particularly for a relatively small language modelling dataset.

### 3. Limited context length

The models used a context window of 30 tokens.

A short context window limits the amount of preceding information available to the model and may prevent it from capturing longer-range relationships.

### 4. Computational constraints

Training was performed in a resource-constrained environment.

In the observed Google Colab CPU runs:

- LSTM training took approximately nine hours for seven epochs.
- Transformer training took approximately two hours per epoch.

These training costs limited the number of experiments and hyperparameter configurations that could reasonably be explored.

Training times depend on the hardware and runtime environment.

### 5. Overfitting

Some models showed a difference between training and validation performance.

The Transformer, for example, continued reducing training loss while validation loss began increasing after its best validation epoch.

Early stopping was used to limit unnecessary training.

### 6. Hyperparameter tuning did not improve performance

Increasing Transformer dropout from 0.1 to 0.2 did not improve the best validation loss.

Further experimentation was limited by computational cost.

### 7. Text generation quality

Greedy generation sometimes produced repetitive sequences.

Temperature sampling increased variation but did not consistently improve grammatical structure or contextual relevance.

### 8. Evaluation limitations

The full classification report and ROC-AUC evaluation were not completed.

With approximately 28,912 possible output classes, a complete classification report would be very large, while calculating full one-vs-rest ROC-AUC would require substantial memory and computation.

An attempt to retain prediction probabilities for the entire test set also caused Google Colab memory issues.

The final evaluation therefore focused on test loss, accuracy, perplexity, weighted precision, weighted recall, and weighted F1-score.

### 9. No production deployment

The project includes lightweight model saving, loading, and inference.

It does not include a production API, web application, Docker container, cloud deployment, or production monitoring.

## Future Improvements

The following improvements would be valuable in a future version of this project.

### Model and architecture improvements

- Experiment with larger Transformer architectures where computational resources permit.
- Explore pretrained language models rather than relying exclusively on models trained from scratch.
- Investigate improved embedding strategies and subword tokenisation, such as Byte Pair Encoding (BPE).
- Experiment with longer context windows.
- Explore learning-rate scheduling and more systematic hyperparameter optimisation.
- Investigate weight decay, dropout, and other regularisation techniques.
- Compare encoder-decoder and decoder-only language modelling approaches where appropriate.

### Dataset improvements

- Experiment with larger and more diverse text datasets.
- Investigate data quality and vocabulary coverage.
- Compare word-level and subword-level tokenisation.
- Explore strategies for handling rare and unknown tokens.

### Evaluation improvements

- Track validation and test perplexity across different model configurations.
- Evaluate top-k next-token accuracy.
- Analyse prediction quality for frequent and rare tokens separately.
- Investigate generation quality using more suitable language-generation evaluation methods and human assessment.
- Explore more memory-efficient evaluation techniques if full probability-based metrics are required.

### Text generation improvements

- Experiment with top-k and nucleus (top-p) sampling.
- Investigate repetition penalties.
- Compare greedy decoding with different sampling temperatures.
- Improve the handling of unknown tokens and generated sequences.

### Deployment improvements

- Build a simple inference interface.
- Develop a FastAPI endpoint for next-token prediction.
- Containerise the application using Docker.
- Explore cloud deployment.
- Introduce model versioning and basic inference monitoring.

These are proposed improvements, not features already implemented in this project.

## Project Structure

```text
Next-Word-Prediction-NLP/
│
├── next_word_prediction.ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
└── colab_models/
    └── Model checkpoints (stored separately)
```

The model checkpoint directory is excluded from the GitHub repository because the combined checkpoint files occupy approximately 614 MB.

The final Transformer checkpoint is retained separately in Google Drive.

To run the inference section without retraining, the final checkpoint must be made available at the path expected by the notebook.

## Technologies Used

- Python
- PyTorch
- Hugging Face Datasets
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Google Colab
- Git and GitHub

## How to Run

1. Clone or download this repository.
2. Open `next_word_prediction.ipynb` in Google Colab or Jupyter Notebook.
3. Install the required dependencies.
4. Run the notebook cells in order.
5. The WikiText-2 dataset will be loaded through Hugging Face Datasets.
6. Run the training and evaluation sections to reproduce the experiments.

**Important:** Reproducing all training experiments can be computationally expensive, particularly on CPU. The trained checkpoints are not included in this repository.

For inference without retraining, obtain the separately stored `final_transformer.pth` checkpoint and place it in the expected `colab_models` directory.

## Key Learnings

Through this project, I gained practical experience in:

- NLP text preprocessing and tokenisation.
- Vocabulary construction and numerical sequence preparation.
- Building neural networks with PyTorch.
- Understanding recurrent neural network architectures.
- Understanding attention mechanisms and Transformers.
- Implementing causal masking and positional embeddings.
- Training, validating, and evaluating language models.
- Applying early stopping and experimenting with hyperparameters.
- Interpreting loss, accuracy, precision, recall, F1-score, and perplexity.
- Implementing next-word prediction and basic text generation.
- Understanding the practical effects of computational limitations and model complexity.

## Conclusion

This project was an important step in developing my understanding of Natural Language Processing and deep learning.

Although the final model achieved modest predictive performance, the project provided practical experience implementing, training, comparing, and evaluating four different neural network architectures.

It also highlighted the challenges of language modelling, including vocabulary size, computational requirements, overfitting, prediction uncertainty, and text generation quality.

The project establishes a foundation for further exploration of pretrained Transformers, large language models, Retrieval-Augmented Generation (RAG), and AI applications.

---

**Project:** Next-Word Prediction Language Model using PyTorch  
**Dataset:** WikiText-2  
**Framework:** PyTorch  
**Task:** Next-token prediction  
**Status:** Educational project completed; future improvements identified.
