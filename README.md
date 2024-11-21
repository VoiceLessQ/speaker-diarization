# Audio Processing Pipeline with Diarization

This repository provides a Python-based pipeline for downloading audio, converting it to a suitable format, performing speaker diarization, and organizing the results. The pipeline is customizable, supports GPU acceleration, and includes integration with YouTube for downloading audio files directly.

---

## Features

- **Audio Conversion:** Converts audio files (e.g., `.mp3`, `.ogg`, `.m4a`) into `.wav` format for processing.
- **Speaker Diarization:** Identifies and segments audio by speakers using the PyAnnote library.
- **YouTube Integration:** Downloads audio from YouTube videos for processing.
- **Output Organization:** Saves speaker-specific audio segments into organized folders.
- **Google Drive Integration:** Uploads processed files to Google Drive for easy access and sharing (optional).

---

## Prerequisites

### Dependencies
- Python 3.8+
- Required Python libraries:
  - `pyannote.audio`
  - `pydub`
  - `pytube`
  - `tqdm`
  - `pydrive2` (optional for Google Drive integration)

### Hardware Requirements
- **GPU Acceleration:** Supports CUDA-enabled GPUs for faster processing.
- Ensure CUDA and cuDNN are properly installed if using GPU.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
