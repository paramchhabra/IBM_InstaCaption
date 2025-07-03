# Instagram Caption Generator

A simple and effective **Instagram Caption Generator** application built using Streamlit. This tool helps users craft engaging and creative captions for their Instagram posts by leveraging the power of language models.

## Features

* Generate creative captions for photos or topics.
* Customize tone and style (funny, motivational, professional, etc.).
* Instant generation using a clean and minimal interface.
* Runs entirely on your local machine.

## Installation

### Install UV Package Manager

Install UV by following the instructions here: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### Initialize UV in the Project Directory

```bash
uv init
```

### Create `.env` File

Add your GROQ\_API\_KEY:

```
GROQ_API_KEY=your_api_key_here
```

### Create a Virtual Environment

```bash
uv venv
```

### Activate the Virtual Environment

On Windows:

```bash
.venv/Scripts/activate
```

On Unix/Mac:

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
uv add -r requirements.txt
```

### Run the Application

```bash
uv run streamlit run app.py
```

Visit the app on your browser using the localhost URL provided by Streamlit, e.g., `http://localhost:8501`

## Usage

1. Launch the app using the above command.
2. Upload the photo you want the caption for.
3. The system will generate a caption, the user can alter the caption acc to their demands.

## Screenshots

