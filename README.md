
# Hugging Face Experiments 🤗

This repository contains my hands-on experiments with Hugging Face and pretrained NLP models using Python.

I explored how pretrained models can be used in real applications, starting with sentiment analysis and understanding how text can be classified as positive or negative.

## 🚀 What I Worked On

- **Hugging Face Transformers:** Used the `transformers` library to work with pretrained models.
- **Sentiment Analysis:** Built a basic text classification pipeline to identify positive and negative sentiment.
- **Confidence Scores:** Explored the scores returned by the model for its predictions.
- **Python Integration:** Experimented with running the model through Python code.
- **RidePulse Connection:** Explored how sentiment analysis could complement RidePulse, my ride-demand prediction project, by analyzing user feedback about ride experiences.

## 🛠️ Tech Stack

- Python
- Hugging Face Transformers
- Pretrained NLP Models
- Gradio (for experimenting with simple interfaces)

## 💻 Getting Started

Clone the repository:

```bash
git clone <your-repository-url>
cd <repository-folder>
```

Install the required libraries:

```bash
pip install transformers torch
```

Run the sentiment analysis experiment:

```python
from transformers import pipeline

sentiment = pipeline("sentiment-analysis")

text = "The ride was comfortable and arrived on time."
result = sentiment(text)

print(result)
```

Example output:

```python
[{'label': 'POSITIVE', 'score': 0.99}]
```

*The score is illustrative; actual results may vary.*

## 🎯 What I Learned

This experience helped me understand how pretrained models can be integrated into Python applications and how NLP can be applied to real-world problems.

I'm continuing to explore AI tools through practical experiments and projects rather than limiting myself to theory.



---

**Learning by building, one experiment at a time.** 🚀

