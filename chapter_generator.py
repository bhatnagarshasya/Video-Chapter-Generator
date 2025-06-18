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
import re
import yt_dlp
from exceptions import (
    VideoProcessingError,
    AudioExtractionError,
    TranscriptionError,
    ExportError
)
from progress import ProgressTracker, track_progress
from nltk.corpus import stopwords

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
        
    def _is_youtube_url(self, url: str) -> bool:
        """Check if the given string is a YouTube URL."""
        youtube_patterns = [
            r'^https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+',
            r'^https?://(?:www\.)?youtube\.com/shorts/[\w-]+',
            r'^https?://youtu\.be/[\w-]+'
        ]
        return any(re.match(pattern, url) for pattern in youtube_patterns)

    def _download_youtube_video(self, url: str) -> str:
        """Download a YouTube video and return the local path."""
        try:
            ydl_opts = {
                'format': 'best[ext=mp4]',
                'outtmpl': 'temp_video_%(id)s.%(ext)s',
                'quiet': True,
                'no_warnings': True
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_path = f"temp_video_{info['id']}.mp4"
                return video_path
        except Exception as e:
            raise VideoProcessingError(f"Failed to download YouTube video: {str(e)}")

    @track_progress(total_steps=3, description="Processing video")
    def process_video(self, video_path: str, progress_tracker: Optional[ProgressTracker] = None) -> List[Dict[str, Any]]:
        """
        Process a video file or YouTube URL and generate chapters.
        
        Args:
            video_path (str): Path to the video file or YouTube URL
            progress_tracker (Optional[ProgressTracker]): Progress tracker for the operation
            
        Returns:
            List[Dict[str, Any]]: List of chapter dictionaries with timestamps and titles
            
        Raises:
            VideoProcessingError: If video processing fails
            AudioExtractionError: If audio extraction fails
            TranscriptionError: If transcription fails
        """
        print("Starting video processing...")
        is_youtube = self._is_youtube_url(video_path)
        temp_video_path = None

        try:
            if is_youtube:
                print("Downloading YouTube video...")
                progress_tracker.set_description("Downloading YouTube video")
                video_path = self._download_youtube_video(video_path)
                temp_video_path = video_path
                progress_tracker.update(1)
                print("YouTube video downloaded successfully")
            elif not os.path.exists(video_path):
                raise VideoProcessingError(f"Video file not found: {video_path}")
            
            print("Extracting audio...")
            progress_tracker.set_description("Extracting audio")
            audio_path = self._extract_audio(video_path)
            progress_tracker.update(1)
            print("Audio extraction completed")
            
            print("Starting transcription...")
            progress_tracker.set_description("Transcribing audio")
            transcription = self._transcribe_audio(audio_path)
            progress_tracker.update(1)
            print("Transcription completed")
            
            print("Generating chapters...")
            progress_tracker.set_description("Generating chapters")
            chapters = self._generate_chapters(transcription)
            progress_tracker.update(1)
            print(f"Generated {len(chapters)} chapters")
            
            print("Cleaning up temporary files...")
            # Clean up temporary files
            if os.path.exists(audio_path):
                os.remove(audio_path)
                print("Removed temporary audio file")
            if temp_video_path and os.path.exists(temp_video_path):
                os.remove(temp_video_path)
                print("Removed temporary video file")
            
            print("Processing completed successfully")
            return chapters
            
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            # Clean up temporary files in case of error
            if temp_video_path and os.path.exists(temp_video_path):
                os.remove(temp_video_path)
                print("Cleaned up temporary video file after error")
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
            print("Loading audio into Whisper...")
            result = self.whisper_model.transcribe(audio_path)
            print("Whisper transcription completed")
            return result
        except Exception as e:
            print(f"Transcription error: {str(e)}")
            raise TranscriptionError(f"Failed to transcribe audio: {str(e)}")
    
    def _generate_chapters(self, transcription: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate chapters from transcription with minimum length enforcement and a maximum of 20 chapters."""
        print("Starting chapter generation process...")
        segments = transcription["segments"]
        if not segments:
            print("No segments found in transcription")
            return []
            
        print(f"Processing {len(segments)} segments...")
        chapters = []
        MIN_CHAPTER_LENGTH = 60  # Minimum chapter length in seconds
        MAX_CHAPTERS = 20
        
        # Group segments into potential chapters
        print("Grouping segments into initial chapters...")
        current_chapter = {
            "start_time": segments[0]["start"],
            "end_time": segments[0]["end"],
            "text": segments[0]["text"],
            "segments": [segments[0]]
        }
        
        for i, segment in enumerate(segments[1:], 1):
            if i % 10 == 0:  # Progress update every 10 segments
                print(f"Processing segment {i}/{len(segments)}...")
            
            # Check for topic transition and minimum length
            if (self._is_topic_transition(current_chapter["text"], segment["text"]) and 
                (segment["start"] - current_chapter["start_time"]) >= MIN_CHAPTER_LENGTH):
                
                # Generate chapter title
                title = self._generate_chapter_title(current_chapter["text"])
                chapters.append({
                    "start_time": current_chapter["start_time"],
                    "end_time": current_chapter["end_time"],
                    "title": title,
                    "description": self._generate_description(current_chapter["text"]),
                    "text": current_chapter["text"]
                })
                
                current_chapter = {
                    "start_time": segment["start"],
                    "end_time": segment["end"],
                    "text": segment["text"],
                    "segments": [segment]
                }
            else:
                current_chapter["end_time"] = segment["end"]
                current_chapter["text"] += " " + segment["text"]
                current_chapter["segments"].append(segment)
        
        print(f"Initial chapter count: {len(chapters)}")
        
        # Add the last chapter if it meets minimum length
        if current_chapter and (current_chapter["end_time"] - current_chapter["start_time"]) >= MIN_CHAPTER_LENGTH:
            title = self._generate_chapter_title(current_chapter["text"])
            chapters.append({
                "start_time": current_chapter["start_time"],
                "end_time": current_chapter["end_time"],
                "title": title,
                "description": self._generate_description(current_chapter["text"]),
                "text": current_chapter["text"]
            })
        
        print("Merging short chapters...")
        # Merge very short chapters with adjacent ones
        if len(chapters) > 1:
            merged_chapters = []
            i = 0
            while i < len(chapters):
                current = chapters[i]
                
                # If this is the last chapter or next chapter is long enough, keep it as is
                if i == len(chapters) - 1 or (chapters[i + 1]["end_time"] - chapters[i + 1]["start_time"]) >= MIN_CHAPTER_LENGTH:
                    merged_chapters.append(current)
                    i += 1
                    continue
                
                # Merge with next chapter if current is too short
                if (current["end_time"] - current["start_time"]) < MIN_CHAPTER_LENGTH:
                    next_chapter = chapters[i + 1]
                    print(f"Merging chapters {i+1} and {i+2}...")
                    merged_chapter = {
                        "start_time": current["start_time"],
                        "end_time": next_chapter["end_time"],
                        "text": current["text"] + " " + next_chapter["text"],
                        "title": self._generate_chapter_title(current["text"] + " " + next_chapter["text"]),
                        "description": self._generate_description(current["text"] + " " + next_chapter["text"])
                    }
                    merged_chapters.append(merged_chapter)
                    i += 2
                else:
                    merged_chapters.append(current)
                    i += 1
            
            chapters = merged_chapters
        
        print(f"Chapters after merging shorts: {len(chapters)}")
        
        # If there are more than MAX_CHAPTERS, merge chapters to fit the limit
        if len(chapters) > MAX_CHAPTERS:
            print(f"Merging chapters to fit maximum limit of {MAX_CHAPTERS}...")
            while len(chapters) > MAX_CHAPTERS:
                # Find the pair of adjacent chapters with the shortest combined duration
                min_duration = float('inf')
                min_idx = 0
                for i in range(len(chapters) - 1):
                    duration = chapters[i + 1]["end_time"] - chapters[i]["start_time"]
                    if duration < min_duration:
                        min_duration = duration
                        min_idx = i
                # Merge the pair
                print(f"Merging chapters {min_idx+1} and {min_idx+2} to fit limit...")
                merged = {
                    "start_time": chapters[min_idx]["start_time"],
                    "end_time": chapters[min_idx + 1]["end_time"],
                    "text": chapters[min_idx]["text"] + " " + chapters[min_idx + 1]["text"],
                    "title": self._generate_chapter_title(chapters[min_idx]["text"] + " " + chapters[min_idx + 1]["text"]),
                    "description": self._generate_description(chapters[min_idx]["text"] + " " + chapters[min_idx + 1]["text"])
                }
                chapters = chapters[:min_idx] + [merged] + chapters[min_idx + 2:]
        
        print(f"Final chapter count: {len(chapters)}")
        
        # Remove the 'text' field from the final output for cleanliness
        for chapter in chapters:
            chapter.pop("text", None)
        
        print("Chapter generation completed")
        return chapters
    
    def _is_topic_transition(self, text1: str, text2: str) -> bool:
        """Detect if there's a topic transition between two text segments."""
        # Simple implementation - can be enhanced with more sophisticated NLP
        sentences1 = sent_tokenize(text1)
        sentences2 = sent_tokenize(text2)
        
        if not sentences1 or not sentences2:
            return False
            
        # Check for transition words and significant content change
        transition_words = [
            "now", "next", "moving on", "let's talk about", "finally",
            "first", "second", "third", "lastly", "in conclusion",
            "to begin", "to start", "let's begin", "let's start",
            "moving forward", "on the other hand", "however",
            "in addition", "furthermore", "moreover", "besides",
            "as a result", "therefore", "thus", "consequently"
        ]
        
        last_sentence = sentences1[-1].lower()
        first_sentence = sentences2[0].lower()
        
        # Check for transition words
        has_transition_word = any(word in last_sentence or word in first_sentence 
                                for word in transition_words)
        
        # Check for significant content change
        content_change = self._calculate_similarity(last_sentence, first_sentence) < 0.3
        
        # Check for time-based transitions (e.g., "after 5 minutes", "in the next hour")
        time_indicators = ["minute", "hour", "second", "time", "period", "duration"]
        has_time_indicator = any(word in last_sentence or word in first_sentence 
                               for word in time_indicators)
        
        return has_transition_word or content_change or has_time_indicator
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text segments."""
        # Simple implementation - can be enhanced with better NLP techniques
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0
    
    def _generate_chapter_title(self, text: str) -> str:
        """Generate a concise, meaningful chapter title from text that is not just the first words of the description, and is never a stopword or too short."""
        import re
        from nltk.corpus import stopwords
        text = text.strip()
        sentences = sent_tokenize(text)
        if not sentences:
            return "Chapter"
        stopword_set = set(stopwords.words('english'))
        generic_words = set([
            'so', 'but', 'and', 'right', 'why', 'they', 'guru', 'because', 'then', 'now', 'yes', 'no', 'well', 'okay', 'sure', 'maybe', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'
        ])
        try:
            # Use the summarizer to generate a headline-style title
            prompt = "Headline: " + text
            summary = self.summarizer(prompt, max_length=15, min_length=4, do_sample=False)[0]["summary_text"]
            summary = summary.strip().rstrip('.:;,-"\' )').lstrip('"\' )')
            # Remove any quotes or incomplete phrases
            summary = re.sub(r'^"|"$', '', summary)
            summary = re.sub(r'^[^a-zA-Z0-9]+', '', summary)
            words = summary.split()
            # Remove leading/trailing stopwords or generic words
            while words and (words[0].lower() in stopword_set or words[0].lower() in generic_words):
                words = words[1:]
            while words and (words[-1].lower() in stopword_set or words[-1].lower() in generic_words):
                words = words[:-1]
            # If the summary is too long, truncate to 12 words
            if len(words) > 12:
                words = words[:12]
            # If the summary is too short or generic, fallback
            if len(words) < 4 or not any(w[0].isalpha() for w in words):
                # Try summarizing the first two sentences
                combined = " ".join(sentences[:2])
                prompt2 = "Headline: " + combined
                combined_summary = self.summarizer(prompt2, max_length=15, min_length=4, do_sample=False)[0]["summary_text"]
                combined_summary = combined_summary.strip().rstrip('.:;,-"\' )').lstrip('"\' )')
                combined_summary = re.sub(r'^"|"$', '', combined_summary)
                combined_summary = re.sub(r'^[^a-zA-Z0-9]+', '', combined_summary)
                combined_words = combined_summary.split()
                while combined_words and (combined_words[0].lower() in stopword_set or combined_words[0].lower() in generic_words):
                    combined_words = combined_words[1:]
                while combined_words and (combined_words[-1].lower() in stopword_set or combined_words[-1].lower() in generic_words):
                    combined_words = combined_words[:-1]
                if len(combined_words) >= 4 and any(w[0].isalpha() for w in combined_words):
                    return " ".join([w.capitalize() for w in combined_words])
                # Fallback: use a template headline
                fallback = f"Main Theme: {sentences[0].split()[0].capitalize() if sentences[0].split() else 'Chapter'}"
                return fallback
            # Capitalize each word for readability
            return " ".join([w.capitalize() for w in words])
        except Exception:
            fallback = f"Key Lesson: {sentences[0].split()[0].capitalize() if sentences[0].split() else 'Chapter'}"
            return fallback
    
    def _generate_description(self, text: str) -> str:
        """Generate a detailed, multi-sentence description for the chapter."""
        text = text.strip()
        if len(text) < 100:
            return text
        try:
            # Try to get a comprehensive, multi-sentence summary
            summary = self.summarizer(
                text,
                max_length=250,
                min_length=100,
                do_sample=False,
                num_beams=5,
                length_penalty=2.5,
                early_stopping=True
            )[0]["summary_text"]
            # If the summary is too short, add more context
            if len(summary) < 150:
                sentences = sent_tokenize(text)
                if len(sentences) > 1:
                    additional_context = " ".join(sentences[1:3])
                    summary = summary + " " + additional_context
            # Remove duplicate sentences
            sentences = sent_tokenize(summary)
            unique_sentences = []
            for sentence in sentences:
                if sentence not in unique_sentences:
                    unique_sentences.append(sentence)
            summary = " ".join(unique_sentences)
            # Ensure the description is not too long
            if len(summary) > 300:
                summary = summary[:297] + "..."
            return summary
        except Exception:
            # Fallback: use the first 2-3 sentences
            sentences = sent_tokenize(text)
            return " ".join(sentences[:3]) if len(sentences) >= 3 else text[:300].strip()
    
    def export_youtube_format(self, chapters: List[Dict[str, Any]], output_path: str):
        """Export chapters in YouTube format."""
        try:
            print(f"Exporting YouTube format to {output_path}...")
            with open(output_path, 'w', encoding='utf-8') as f:
                for chapter in chapters:
                    timestamp = self._format_timestamp(chapter['start_time'])
                    f.write(f"{timestamp} {chapter['title']}\n")
            print("YouTube format export completed")
        except Exception as e:
            print(f"Export error: {str(e)}")
            raise ExportError(f"Failed to export YouTube format: {str(e)}")
    
    def export_srt(self, chapters: List[Dict[str, Any]], output_path: str):
        """Export chapters in SRT format."""
        try:
            print(f"Exporting SRT format to {output_path}...")
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, chapter in enumerate(chapters, 1):
                    start_time = self._format_timestamp(chapter['start_time'], srt=True)
                    end_time = self._format_timestamp(chapter['end_time'], srt=True)
                    f.write(f"{i}\n{start_time} --> {end_time}\n{chapter['title']}\n{chapter['description']}\n\n")
            print("SRT format export completed")
        except Exception as e:
            print(f"Export error: {str(e)}")
            raise ExportError(f"Failed to export SRT format: {str(e)}")
    
    def export_json(self, chapters: List[Dict[str, Any]], output_path: str):
        """Export chapters in JSON format."""
        try:
            print(f"Exporting JSON format to {output_path}...")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chapters, f, indent=2)
            print("JSON format export completed")
        except Exception as e:
            print(f"Export error: {str(e)}")
            raise ExportError(f"Failed to export JSON format: {str(e)}")
    
    def _format_timestamp(self, seconds: float, srt: bool = False) -> str:
        """Format seconds into SRT timestamp format."""
        if srt:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            seconds = seconds % 60
            milliseconds = int((seconds - int(seconds)) * 1000)
            
            return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            seconds = seconds % 60
            milliseconds = int((seconds - int(seconds)) * 1000)
            
            return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}" 