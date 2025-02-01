# Context-Aware Code Reviewer

An autonomous code review assistant that uses AI to provide context-aware feedback and suggestions.

## Overview

This project aims to create an automated code review assistant using modern AI techniques.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the API server:
```bash
uvicorn src.review_assistant.api:app --reload
```

## API Endpoints

- `GET /` - Health check
- `POST /review` - Submit code for review
