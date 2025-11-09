# Chat with Various type Documents using Meta AI

This Streamlit application allows you to ask questions about scientific documents and receive answers based on the content. It utilizes text extraction from various file formats and leverages the power of a pre-trained language model for information retrieval.

## Features:

* Extract text from PDF, DOCX, HTML, and image files.
* Process extracted text into meaningful chunks.
* Handle user questions and provide answers based on the document content.
* Maintain a chat history for previous interactions.

## Note:

This application relies on the `meta_ai_api` library, which serves as a mock for a real Meta AI API. Thanks to the work of [@Strvm](https://github.com/Strvm) at [Strvm/meta-ai-api](https://github.com/Strvm/meta-ai-api) on this library, we don't require an actual Meta AI API key.

## Requirements:

Please refer to the `requirements.txt` file for a list of dependencies needed to run this application.

## Installation
1) Clone the repository:
   
```git clone https://github.com/CodewithAbhi7/Chat-with-Various-type-Documents-using-Meta-AI```

2) Install the required dependencies:

```pip install -r requirements.txt```

## Running the Application
Start the Streamlit application with the following command:


```streamlit run test.py```

## Usage:

1. Start the application.
2. Select the file types you want to process (PDF, DOCX, HTML, Image).
3. Upload your documents.
4. Click the "Process" button to extract text.
5. Once processed, ask a question about the document content in the text box.
6. Click "Enter" or submit the question.
7. The application will display an answer based on the extracted text and the pre-trained model.

## Contributing:

We welcome contributions to this project! Feel free to fork the repository and submit pull requests with your   
 enhancements.
