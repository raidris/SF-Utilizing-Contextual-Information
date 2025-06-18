# Intent Detection and Slot Filling: Improving Slot Filling by Utilizing Contextual Information

---

## Abstract

Development and evaluation of a Natural Language Understanding (NLU) system, leveraging deep learning to achieve intent detection and slot filling from user queries. Utilizing the ATIS and SNIPS datasets, the project explores advanced techniques such as adversarial training for Mutual Information Maximization, POS tag embeddings, and sequence modeling through LSTM networks and CRF layers.

---

## Introduction

The advancement of dialogue systems and virtual assistants has necessitated sophisticated NLU systems capable of interpreting user intentions and extracting relevant entities. This project aims to develop a robust NLU model to accurately perform intent detection and slot filling, crucial components for understanding user queries within a dialogue system.

---

## Motivation

Understanding natural language is a cornerstone of artificial intelligence, enabling machines to interpret, respond to, and act upon human languages. A critical sub-task in this domain is Slot Filling (SF), vital for spoken language understanding (SLU) systems. SF involves identifying semantic constituents within a given natural language utterance, which is crucial for developing efficient personal assistant tools such as Siri, Alexa, and Google Assistant.

Therefore, I decided to take on the challenge of understanding this method, by finding a paper, and essentially, implementing the methods and technologies discussed in it.

---

## The Paper

My work is inspired by the paper *"Improving Slot Filling by Utilizing Contextual Information,"* which addresses the significance of contextual information in SF tasks. Traditional models have either overlooked the potential of contextual data or employed it in a restricted manner, such as through self-attention mechanisms. This study introduces a novel approach that integrates contextual information on two fronts: the representation level and the task-specific (i.e., label) level. By incorporating mutual information (MI) maximization and introducing auxiliary tasks for context-based word and sentence-level predictions, My model achieves state-of-the-art performance across three benchmark datasets.

---

## Methodology

### Data Preparation

The project employed the ATIS and SNIPS datasets, annotating user queries with intents and slots. To enrich the model's input, POS tags were generated for each sentence, providing additional syntactic information.

I found the datasets on huggingface.co, but noticed that the SNIPS dataset lacked slots. I located a different implementation with pre-processed datasets and used those after verification.

Each dataset (ATIS and SNIPS) was split into train, dev, and test sets, each containing three files:
- `label`: Intent labels
- `seq.in`: Word sequences
- `seq.out`: Slot tags

Each row represents one sentence's data.

### Data Preprocessing

I used the Natural Language Toolkit (NLTK) to generate POS tags for each sentence. A special `<PAD>` token was included to handle variable sentence lengths. This step prepared the data for neural network models.

### Custom Dataset Class

The `DatasetObj` class converts sentences, slot tags, POS tags, and labels into indexed tensors. A custom `collate_fn` ensures proper padding for batching during training and evaluation.

---

## Model Architecture

My model follows the framework of the referenced article, incorporating:

- **Word and POS Tag Embeddings:** Pre-trained word embeddings (e.g., GloVe) and learned POS embeddings.
- **Bi-directional LSTM (BiLSTM):** For capturing contextual nuances.
- **CRF Layer:** For improved slot tagging accuracy.
- **Adversarial Training:** Enhances MI between word embeddings and context using a discriminator.
- **Sentence-Level Prediction:** Supports intent detection.
- **Multi-Task Learning:** Adds auxiliary tasks for better contextual understanding.

---

## Training and Evaluation

I tuned hyperparameters including learning rates, batch sizes, and architecture dimensions. CRF was used for slot filling, binary cross-entropy for intent detection, and mutual information maximization guided the training strategy.

---

## Results

![image](https://github.com/user-attachments/assets/3dfb7cd5-4b4d-412f-b80d-997254369712)

---
