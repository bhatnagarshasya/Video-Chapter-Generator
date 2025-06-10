# Video Chapter Generator

An intelligent system that automatically generates chapter markers for long videos using speech recognition and content analysis.

## Features

- **Video Analysis**
  - Audio extraction
  - Speech recognition using OpenAI's Whisper
  - Topic segmentation
  - Transition detection

- **Chapter Generation**
  - Meaningful chapter titles
  - Accurate timestamps
  - Description writing
  - Duration optimization

- **Export Options**
  - YouTube format
  - Video editor markers
  - SRT/VTT files
  - JSON metadata

## Prerequisites

- Python 3.8+
- FFmpeg (for video processing)
- CUDA-compatible GPU (recommended for faster processing)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/video-chapter-generator.git
cd video-chapter-generator
```

2. Create and activate a virtual environment:

For Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
```

For Linux/Mac:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install FFmpeg:
   - Windows: Download from https://www.gyan.dev/ffmpeg/builds/ and add to PATH
   - Or use package manager:
     ```bash
     # Using Chocolatey
     choco install ffmpeg
     
     # Using winget
     winget install ffmpeg
     ```

5. Download required NLTK data:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('punkt_tab')"
```

## Usage

1. Place your video file in the project directory. The video should be in a common format (MP4, MKV, AVI, etc.).

2. Update the video path in `example.py`:
```python
# Change this line in example.py
video_path = "your_video.mp4"  # Replace with your video filename
```

3. Run the script:
```bash
python example.py
```

The script will:
- Extract audio from your video
- Transcribe the audio using Whisper
- Generate chapters based on content analysis
- Create three output files:
  - `chapters_youtube.txt`: YouTube chapter format
  - `chapters.srt`: SubRip subtitle format
  - `chapters.json`: JSON metadata with full chapter information

## Output Files

1. **YouTube Format** (`chapters_youtube.txt`):
```
0:00:00 Introduction
0:05:30 Main Topic
0:15:45 Conclusion
```

2. **SRT Format** (`chapters.srt`):
```
1
00:00:00,000 --> 00:05:30,000
Introduction
Chapter description...

2
00:05:30,000 --> 00:15:45,000
Main Topic
Chapter description...
```

3. **JSON Format** (`chapters.json`):
```json
[
  {
    "start_time": 0,
    "end_time": 330,
    "title": "Introduction",
    "description": "Chapter description..."
  },
  {
    "start_time": 330,
    "end_time": 945,
    "title": "Main Topic",
    "description": "Chapter description..."
  }
]
```

## Performance Notes

- First run will download the Whisper model (about 1.5GB)
- Processing time depends on:
  - Video length
  - Available hardware (CPU/GPU)
  - Model size (tiny/base/small/medium/large)
- For faster processing:
  - Use GPU if available
  - Use smaller model size (e.g., "tiny" or "base")
  - Process shorter videos


## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 
