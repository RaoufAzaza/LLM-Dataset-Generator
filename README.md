The LLM Dataset Generator is a Streamlit-based tool that allows users to generate question-answer (QA) pairs from PDF documents using large language models (LLMs). This project leverages models like Llama and Mistral to transform PDF content into structured QA datasets, which can then be downloaded in various formats (CSV, JSONL) for use in machine learning applications.

Features

PDF Parsing: Extracts and cleans text from PDF documents for further processing.


QA Pair Generation: Creates customized QA pairs based on the parsed PDF content in either English or French.


Dataset Splitting: Splits the generated QA pairs into training and validation sets.


Flexible Output Formats: Supports exporting datasets in both CSV and JSONL formats.


Model Selection: Choose from available models, such as Llama 3.2 (11B) or Mistral (7B), to fit specific requirements.

How It Works


Upload PDF: Start by uploading a PDF document through the Streamlit interface.


Select Model and Language: Choose the desired LLM model and output language for generating QA pairs.


Generate QA Pairs: The tool uses the selected model to generate a specified number of QA pairs based on the document content.


Download Dataset: Download the generated training and validation datasets in the preferred format.
