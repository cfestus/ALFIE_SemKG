---
component-id: ALFIE_SemKG
type: Software
name: The ALFIE Semantic Knowledge Graph pipeline
description: This repository contains a Python pipeline for Semantic Knowledge Graph construction from TQAVE-structured AI Ethics and Governance discourse, comprising themes, questions, answers, votes, and expert contributions. The code processes semi-structured discourse data, extracts entities and relations with GPT API, maps them to an ontology-aligned representation, and generates provenance-aware RDF knowledge graphs. It also supports deterministic IRI minting, canonicalisation, evidence linking, and SHACL validation
work-package: 
  - WP3
pilot:
  - Semantic Knowledge Graph
project: alfie-project
release-date: 18/03/2026
release-number: v1.0
licence:
  - MIT license
copyright: "Copyright (c) 2026 ALFIE"
contributors:
  - Chukwudi "Festus" Uwasomba <https://github.com/cfestus>
  - Nonso Nnamoko
  - Yannis Korkontzelos
credits:
  - https://github.com/cfestus
---

## Prerequisites
- Python 3.x
- OpenAI API key

## To run the code 

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt.

4. **API Configuration**:
Create a .env file in the root directory and add your OpenAI API key, that is your GPT-4 API.
   ```bash
   OPENAI_API_KEY="your_api_key_here"

5. **Usage**:
Once you have completed the setup, run the main script to Semantic Knowledge Graphs.
   ```bash
   run main.py
