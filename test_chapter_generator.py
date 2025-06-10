import unittest
import os
import tempfile
from chapter_generator import VideoChapterGenerator
from exceptions import VideoProcessingError, AudioExtractionError, TranscriptionError

class TestVideoChapterGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = VideoChapterGenerator(model_size="tiny")  # Use tiny model for faster tests
        
    def test_invalid_video_path(self):
        """Test handling of invalid video path."""
        with self.assertRaises(VideoProcessingError):
            self.generator.process_video("nonexistent_video.mp4")
            
    def test_export_formats(self):
        """Test export functionality with sample chapters."""
        chapters = [
            {
                "start_time": 0,
                "end_time": 60,
                "title": "Introduction",
                "description": "Chapter 1 description"
            },
            {
                "start_time": 60,
                "end_time": 120,
                "title": "Main Content",
                "description": "Chapter 2 description"
            }
        ]
        
        # Test YouTube format export
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            self.generator.export_youtube_format(chapters, f.name)
            with open(f.name, 'r') as f2:
                content = f2.read()
                self.assertIn("0:00:00 Introduction", content)
                self.assertIn("0:01:00 Main Content", content)
            os.unlink(f.name)
            
        # Test SRT export
        with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as f:
            self.generator.export_srt(chapters, f.name)
            with open(f.name, 'r') as f2:
                content = f2.read()
                self.assertIn("00:00:00,000 --> 00:01:00,000", content)
                self.assertIn("Introduction", content)
            os.unlink(f.name)
            
        # Test JSON export
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            self.generator.export_json(chapters, f.name)
            with open(f.name, 'r') as f2:
                content = f2.read()
                self.assertIn('"title": "Introduction"', content)
                self.assertIn('"title": "Main Content"', content)
            os.unlink(f.name)
            
    def test_topic_transition(self):
        """Test topic transition detection."""
        # Test with transition words
        text1 = "Now let's move on to the next topic."
        text2 = "This is a new topic."
        self.assertTrue(self.generator._is_topic_transition(text1, text2))
        
        # Test with similar content
        text1 = "The cat sat on the mat."
        text2 = "The cat was comfortable on the mat."
        self.assertFalse(self.generator._is_topic_transition(text1, text2))
        
    def test_chapter_title_generation(self):
        """Test chapter title generation."""
        # Test with short text
        text = "This is a short chapter."
        title = self.generator._generate_chapter_title(text)
        self.assertEqual(title, text)
        
        # Test with long text
        text = "This is a very long chapter that needs to be summarized into a shorter title that captures the main point of the content while being concise and informative for the viewer to understand what this section is about."
        title = self.generator._generate_chapter_title(text)
        self.assertLess(len(title), len(text))
        self.assertGreater(len(title), 0)

if __name__ == '__main__':
    unittest.main() 