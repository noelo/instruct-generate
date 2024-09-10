# Ingest-Splitter and Ingest-Local Pipelines

This repository contains two Jupyter notebooks that form a pipeline for document ingestion, chunking, and language model interaction. These notebooks allow you to convert documents (like PDFs), split them into smaller chunks, and send those chunks to a large language model (such as Mistral-7B Instruct) for processing or text generation.

## Files in the Repo

- **injest-splitter.ipynb**: A notebook designed to send chunks of text to a large language model for completion or generation tasks. It uses the Mistral-7B Instruct model via an API.
  
- **injest-local.ipynb**: A notebook that ingests documents, splits them into chunks, and prepares them for further processing by the language model.

## Prerequisites

- Python 3.10+
- Jupyter Notebook
- Libraries (installed in the notebooks):
  - `docling`
  - `quackling`
  - `llama-index`
  - `semantic-router`
  - `semantic-chunkers`
  - `rich`

## Installation

1. Clone the repository:

   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. Install the required libraries:

   Run the first few code cells in either notebook to install the required dependencies automatically via `%pip`.

## Usage

### 1. Ingest-Local (Document Ingestion and Chunking)

This notebook (`injest-local.ipynb`) is used to ingest and split a document into chunks:

- Load your document (PDF) in the variable `source`.
- The `DocumentConverter` and `DoclingPDFReader` will process the document into a chunked format.
- The chunking process is handled by `RollingWindowSplitter` and `StatisticalChunker`, which store the chunks in the `splits` and `chunks` variables.

You can modify the chunking parameters, including `min_split_tokens` and `max_split_tokens`, to fit your needs.

If you want to save the chunked output, you can write it to a file. Here’s an example of how to save the chunks to a JSON file:

```python
import json

with open('chunked_data.json', 'w') as f:
    json.dump([chunk.to_dict() for chunk in chunks], f)
```

### 2. Ingest-Splitter (Model Interaction)

Once the document is chunked, use the `injest-splitter.ipynb` notebook to send those chunks to a large language model.

- Load the chunked data (either by running the chunking notebook first or loading a previously saved file).
- The notebook will use the OpenLLM library to interact with the model and stream responses for each chunk.
  
You can modify the `max_tokens` and `timeout` settings to control the model's output length and response time.

### Example Flow

1. **Step 1**: Run `injest-local.ipynb` to ingest and chunk the document.
2. **Step 2**: Save the chunked data as a file (optional).
3. **Step 3**: Run `injest-splitter.ipynb` to send the chunks to the language model and receive the responses.

### Environment Variables

Both notebooks require environment variables to connect to the language model API:

- `API_KEY`: Your API key for accessing the model.
- `LLM_URL`: The base URL for the language model API.

You can load these environment variables using a `.env` file or by setting them directly in the notebook.

Example `.env` file:

```
API_KEY=your-api-key
LLM_URL=your-llm-url
```

## Customization

- Adjust the chunking parameters in `injest-local.ipynb` for your document.
- Modify the prompt generation and response handling in `injest-splitter.ipynb` to fit your needs.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.