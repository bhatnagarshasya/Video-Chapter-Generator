# Video Chapter Generator

An intelligent system that automatically generates chapter markers for long videos using speech recognition and content analysis.

## Features

- **Video Analysis**
  - Audio extraction
  - Speech recognition using OpenAI's Whisper
  - Topic segmentation
  - Transition detection

- **Chapter Generation**
  - Headline-style, meaningful chapter titles (not just transcript fragments)
  - Accurate timestamps
  - Multi-sentence, informative chapter descriptions
  - Duration optimization (min/max chapter length, max 20 chapters)

- **Export Options**
  - YouTube format
  - Video editor markers (SRT)
  - SRT/VTT files
  - JSON metadata

## Planned Features (Future Roadmap)

- **Thumbnail Generation:**
  - Automatically select or generate a thumbnail for each chapter based on video content.
- **Preview Generation:**
  - Create short video previews or GIFs for each chapter to enhance navigation and user experience.
- **Visual Scene Detection:**
  - Use computer vision to detect scene changes and further improve chapter segmentation accuracy.
- **Multi-language Support:**
  - Support for non-English videos using Whisper's multilingual capabilities.
- **SEO Optimization:**
  - Generate SEO-friendly chapter titles and descriptions for better discoverability.

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

1. Place your video file in the project directory or use a YouTube link.

2. Run the script:
```bash
python example.py --video "your_video.mp4" --model base
# or
python example.py --youtube "YOUTUBE_URL" --model base
```

### Model Size Options
- The `--model` argument controls the Whisper model size used for transcription and chapter generation.
- Available options:
  - `tiny`   (fastest, least accurate)
  - `base`   (fast, balanced)
  - `small`  (good accuracy, moderate speed)
  - `medium` (high accuracy, slower)
  - `large`  (best accuracy, slowest, most resource-intensive)
- **Tip:** For quick tests or long videos, use `tiny` or `base`. For best accuracy, use `medium` or `large` (requires more RAM/VRAM).

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
0:00:00 Main Theme Of Karma
0:05:30 Understanding Life's Uncertainties
0:15:45 Developing Spiritual Discipline
```

2. **SRT Format** (`chapters.srt`):
```
1
00:00:00,000 --> 00:05:30,000
Main Theme Of Karma
Chapter description...

2
00:05:30,000 --> 00:15:45,000
Understanding Life's Uncertainties
Chapter description...
```

3. **JSON Format** (`chapters.json`):
```json
[
  {
    "start_time": 0,
    "end_time": 330,
    "title": "Main Theme Of Karma",
    "description": "Chapter description..."
  },
  {
    "start_time": 330,
    "end_time": 945,
    "title": "Understanding Life's Uncertainties",
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
