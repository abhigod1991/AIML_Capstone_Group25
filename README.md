# AIML_Capstone_Group25
This repository contains artifacts of AIML Capstone project

**Objective**
1. Explore and utilize available Generative AI models for the machine translation task.
2. Understand Bias Detection in Machine Translation
3. Identify unintended prejudices or stereotypes in translated text.

**Dataset: WinoMT**

1. en.txt: Full corpus
2. Source Languages: Spanish, Hindi 
3. Target Language: English

**Model**
facebook/mbart-large-50-many-to-many-mmt: Pre-trained models from Hugging Face

**Bias Detection Metrics**
1. BLEU Score: Measures translation quality (0 to 1 scale)
2. Classifier: Accuracy of gender prediction in translations
3. Fairness Indicators (AI Fairness 360): Toolkit for detecting and analyzing bias in MT systems

     Statistical Parity Difference (SPD): Measures the difference in the probability of favorable outcomes between two groups. Formula:
	          SPD = P(Outcome|Group A) - P(Outcome|Group B)
      For example, in gender bias detection, a favourable outcome could be accurate gender representation in translations

      Disparate Impact (DI): Measures the ratio of favorable outcomes between two groups. Formula:
	          DI = P(Outcome|Group A) / P(Outcome|Group B)
      This metric evaluates whether one group is disproportionately favoured compared to another.

**Deployed app link:** https://huggingface.co/spaces/Team25/AIML_Capstone_Project (private)
