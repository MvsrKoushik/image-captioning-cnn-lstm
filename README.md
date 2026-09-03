# Image Captioning with CNN + LSTM

The image-captioning Colab reorganized into testable preprocessing and model components. A pretrained CNN encodes images and an LSTM decoder generates captions with explicit BOS/EOS handling.

```bash
pip install -e .[train,dev]
pytest
```

Datasets and checkpoints are intentionally external. Evaluation should report BLEU alongside qualitative examples and the exact data split.

