import os
import json
import whisper
import numpy as np
from moviepy.editor import VideoFileClip
from transformers import pipeline
from nltk.tokenize import sent_tokenize
from datetime import timedelta
from typing import List, Dict, Any, Optional
import torch
from exceptions import (
    VideoProcessingError,
    AudioExtractionError,
    TranscriptionError,
    ExportError
)
from progress import ProgressTracker, track_progress

class VideoChapterGenerator:
    def __init__(self, model_size: str = "base"):
        """
        Initialize the VideoChapterGenerator.
        
        Args:
            model_size (str): Size of the Whisper model to use (tiny, base, small, medium, large)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.whisper_model = whisper.load_model(model_size).to(self.device)
            self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        except Exception as e:
            raise VideoProcessingError(f"Failed to initialize models: {str(e)}")
        
    @track_progress(total_steps=3, description="Processing video")
    def process_video(self, video_path: str, progress_tracker: Optional[ProgressTracker] = None) -> List[Dict[str, Any]]:
        """
        Process a video file and generate chapters.
        
        Args:
            video_path (str): Path to the video file
            progress_tracker (Optional[ProgressTracker]): Progress tracker for the operation
            
        Returns:
            List[Dict[str, Any]]: List of chapter dictionaries with timestamps and titles
            
        Raises:
            VideoProcessingError: If video processing fails
            AudioExtractionError: If audio extraction fails
            TranscriptionError: If transcription fails
        """
        if not os.path.exists(video_path):
            raise VideoProcessingError(f"Video file not found: {video_path}")
            
        try:
            # Extract audio
            progress_tracker.set_description("Extracting audio")
            audio_path = self._extract_audio(video_path)
            progress_tracker.update(1)
            
            # Transcribe audio
            progress_tracker.set_description("Transcribing audio")
            transcription = self._transcribe_audio(audio_path)
            progress_tracker.update(1)
            
            # Generate chapters
            progress_tracker.set_description("Generating chapters")
            chapters = self._generate_chapters(transcription)
            progress_tracker.update(1)
            
            # Clean up temporary files
            if os.path.exists(audio_path):
                os.remove(audio_path)
                
            return chapters
            
        except Exception as e:
            if isinstance(e, (VideoProcessingError, AudioExtractionError, TranscriptionError)):
                raise
            raise VideoProcessingError(f"Failed to process video: {str(e)}")
    
    def _extract_audio(self, video_path: str) -> str:
        """Extract audio from video file."""
        try:
            video = VideoFileClip(video_path)
            audio_path = "temp_audio.wav"
            video.audio.write_audiofile(audio_path)
            video.close()
            return audio_path
        except Exception as e:
            raise AudioExtractionError(f"Failed to extract audio: {str(e)}")
    
    def _transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe audio using Whisper."""
        try:
            result = self.whisper_model.transcribe(audio_path)
            return result
        except Exception as e:
            raise TranscriptionError(f"Failed to transcribe audio: {str(e)}")
    
    def _generate_chapters(self, transcription: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate chapters from transcription."""
        segments = transcription["segments"]
        if not segments:
            return []
            
        chapters = []
        
        # Group segments into potential chapters
        current_chapter = {
            "start_time": segments[0]["start"],
            "end_time": segments[0]["end"],
            "text": segments[0]["text"]
        }
        
        for segment in segments[1:]:
            # Check for topic transition
            if self._is_topic_transition(current_chapter["text"], segment["text"]):
                # Generate chapter title
                title = self._generate_chapter_title(current_chapter["text"])
                chapters.append({
                    "start_time": current_chapter["start_time"],
                    "end_time": current_chapter["end_time"],
                    "title": title,
                    "description": self._generate_description(current_chapter["text"])
                })
                
                current_chapter = {
                    "start_time": segment["start"],
                    "end_time": segment["end"],
                    "text": segment["text"]
                }
            else:
                current_chapter["end_time"] = segment["end"]
                current_chapter["text"] += " " + segment["text"]
        
        # Add the last chapter
        if current_chapter:
            title = self._generate_chapter_title(current_chapter["text"])
            chapters.append({
                "start_time": current_chapter["start_time"],
                "end_time": current_chapter["end_time"],
                "title": title,
                "description": self._generate_description(current_chapter["text"])
            })
        
        return chapters
    
    def _is_topic_transition(self, text1: str, text2: str) -> bool:
        """Detect if there's a topic transition between two text segments."""
        # Simple implementation - can be enhanced with more sophisticated NLP
        sentences1 = sent_tokenize(text1)
        sentences2 = sent_tokenize(text2)
        
        if not sentences1 or not sentences2:
            return False
            
        # Check for transition words and significant content change
        transition_words = ["now", "next", "moving on", "let's talk about", "finally"]
        last_sentence = sentences1[-1].lower()
        first_sentence = sentences2[0].lower()
        
        return any(word in last_sentence for word in transition_words) or \
               self._calculate_similarity(last_sentence, first_sentence) < 0.3
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text segments."""
        # Simple implementation - can be enhanced with better NLP techniques
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0
    
    def _generate_chapter_title(self, text: str) -> str:
        """Generate a meaningful chapter title from text."""
        # Use the first sentence or a summary of the text
        sentences = sent_tokenize(text)
        if not sentences:
            return "Chapter"
            
        # Generate a summary if the text is too long
        if len(text) > 200:
            try:
                summary = self.summarizer(text, max_length=50, min_length=10)[0]["summary_text"]
                return summary
            except Exception:
                return sentences[0]
        return sentences[0]
    
    def _generate_description(self, text: str) -> str:
        """Generate a description for the chapter."""
        if len(text) > 200:
            try:
                return self.summarizer(text, max_length=100, min_length=30)[0]["summary_text"]
            except Exception:
                return text
        return text
    
    def export_youtube_format(self, chapters: List[Dict[str, Any]], output_path: str):
        """Export chapters in YouTube format."""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for chapter in chapters:
                    start_time = str(timedelta(seconds=int(chapter["start_time"])))
                    f.write(f"{start_time} {chapter['title']}\n")
        except Exception as e:
            raise ExportError(f"Failed to export YouTube format: {str(e)}")
    
    def export_srt(self, chapters: List[Dict[str, Any]], output_path: str):
        """Export chapters in SRT format."""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for i, chapter in enumerate(chapters, 1):
                    start_time = self._format_timestamp(chapter["start_time"])
                    end_time = self._format_timestamp(chapter["end_time"])
                    
                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{chapter['title']}\n")
                    f.write(f"{chapter['description']}\n\n")
        except Exception as e:
            raise ExportError(f"Failed to export SRT format: {str(e)}")
    
    def export_json(self, chapters: List[Dict[str, Any]], output_path: str):
        """Export chapters in JSON format."""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(chapters, f, indent=2)
        except Exception as e:
            raise ExportError(f"Failed to export JSON format: {str(e)}")
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds into SRT timestamp format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        milliseconds = int((seconds - int(seconds)) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}" 