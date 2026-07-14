# Reconnaissance d'émotions vocales (RAVDESS)

Classification de l'émotion exprimée dans un enregistrement audio (8 classes) à partir de ses caractéristiques MFCC, avec des CNN implémentés en Keras/TensorFlow.

## Pipeline

1. **Extraction MFCC** — `librosa` (40 coefficients, fenêtre 25ms / décalage 10ms), avec padding/troncature à une longueur temporelle fixe.
2. **Prétraitement** — normalisation min-max, encodage des labels (0-7).
3. **Modélisation** — deux architectures testées :
   - CNN 2D (Conv2D + MaxPooling + Dense) sur le spectrogramme MFCC
   - CNN 1D + GlobalAveragePooling (plus robuste aux variations de durée)

## Résultats

~40% d'accuracy en validation sur 8 classes (hasard : 12,5%). Dataset volontairement petit (1200 exemples d'entraînement), donc performance plafonnée par la quantité de données plutôt que par l'architecture.

## Stack

Python · TensorFlow/Keras · librosa · NumPy · SciPy

## Dataset

[RAVDESS](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) (Ryerson Audio-Visual Database of Emotional Speech and Song)

## Pistes d'amélioration

- Data augmentation (SpecAugment)
- Couches récurrentes (LSTM/GRU) pour capter la dynamique temporelle
- Transfer learning avec des embeddings audio pré-entraînés (wav2vec2, etc.)

