from chapter_generator import VideoChapterGenerator
import os

def main():
    # Initialize the chapter generator
    generator = VideoChapterGenerator(model_size="base")  # Use "base" for faster processing
    
    # Video path - replace "your_video.mp4" with your actual video filename
    video_path = "2025-06-04 21-42-28.mkv"
    
    if not os.path.exists(video_path):
        print(f"Please place your video file at: {video_path}")
        print("Or update the video_path variable in example.py to match your video filename")
        return
    
    print("Processing video...")
    # Generate chapters
    chapters = generator.process_video(video_path)
    
    # Export in different formats
    print("\nExporting chapters...")
    generator.export_youtube_format(chapters, "chapters_youtube.txt")
    generator.export_srt(chapters, "chapters.srt")
    generator.export_json(chapters, "chapters.json")
    
    print("\nGenerated chapters:")
    for i, chapter in enumerate(chapters, 1):
        print(f"\nChapter {i}:")
        print(f"Title: {chapter['title']}")
        print(f"Time: {chapter['start_time']:.2f}s - {chapter['end_time']:.2f}s")
        print(f"Description: {chapter['description'][:100]}...")

if __name__ == "__main__":
    main() 