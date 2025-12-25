# SteganoML
SteganoML is an adaptive machine learning–driven audio steganography
framework that improves robustness and imperceptibility by embedding
encrypted data only into acoustically stable audio frames.

---

## Problem Statement

Traditional LSB-based audio steganography techniques embed data either
sequentially or randomly across audio samples. While simple, such approaches
are highly vulnerable to noise, compression, and signal processing attacks,
often leading to perceptible distortion or data loss.

SteganoML addresses these limitations by using machine learning to identify
perceptually and statistically stable audio frames and selectively embed
encrypted payloads only within those regions.

---

## Methodology Overview

SteganoML follows a multi-stage adaptive pipeline:

- Audio is segmented into frames and acoustic features are extracted
  (MFCCs, zero-crossing rate, energy, spectral features, etc.).
- A supervised CatBoost model predicts which frames are suitable for
  secure data embedding.
- The secret payload is encrypted using AES-256, with keys derived
  using PBKDF2.
- Adaptive LSB embedding is applied only to ML-selected stable frames,
  improving robustness and imperceptibility.

---

## Repository Structure

- `src/` – Core implementation and evaluation scripts  
- `models/` – Pretrained CatBoost model and feature scaler  
- `results/` – Experimental results, plots, and evaluation metrics  
- `paper/` – Final research paper  

---

## Experimental Results

SteganoML was evaluated against a randomized LSB baseline using
PSNR, SNR, BER, and normalized correlation metrics.

The results demonstrate improved robustness and imperceptibility while
intentionally sacrificing payload capacity to prioritize audio quality
and security, as discussed in the paper.

---

## Research
We explore whether interpretable, rule-based ML models can provide competitive steganographic performance while offering explainability guarantees absent in deep neural approaches.

---

## Environment Requirements

This project has been tested and verified on **Python 3.10**.

Due to compatibility limitations of certain audio playback libraries
with newer Python versions, users are strongly recommended to use
Python 3.10 for reliable execution.

---
> This work is under review 
