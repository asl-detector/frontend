# Frontend Application

This directory contains the ASL Dataset collector application's frontend components, which enable users to upload, annotate, and process ASL videos.

## Components

- `gui.py` - Main application interface built with PyQt
- `extract.py` - Utilities for extracting features from videos
- `pe_cli.py` - Command-line pose estimation tool
- `md/` - Legal documents and agreements:
  - `data-agreement.md` - Data usage license agreement
  - `privacy.md` - Privacy policy
  - `tos.md` - Terms of service
- `tasks/` - Configuration files for ML tasks

## Usage

The frontend application provides a graphical interface for users to:
- Upload ASL videos
- Annotate videos with text translations
- Process videos using pose estimation models
- View and manage their uploaded content

## Setup

To run the application:

```bash
python gui.py
```

Users will be prompted to accept the terms of service and privacy policy before using the application.
```
# asl-backend
