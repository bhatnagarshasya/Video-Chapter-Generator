from chapter_generator import VideoChapterGenerator
import os
import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError

def process_video(generator, video_path):
    """Process a video and generate chapters."""
    print(f"\nProcessing video: {video_path}")
    try:
        # Set a timeout of 30 minutes (1800 seconds)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(generator.process_video, video_path)
            try:
                # Generate chapters with timeout
                chapters = future.result(timeout=1800)
                
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
            except TimeoutError:
                print("\nError: Processing took too long (timeout after 30 minutes)")
                print("Try using a smaller model size (e.g., --model tiny)")
    except Exception as e:
        print(f"\nError: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Generate video chapters from local video or YouTube URL')
    parser.add_argument('--video', '-v', help='Path to local video file')
    parser.add_argument('--youtube', '-y', help='YouTube video URL')
    parser.add_argument('--model', '-m', default='base', choices=['tiny', 'base', 'small', 'medium', 'large'],
                      help='Whisper model size (default: base)')
    
    args = parser.parse_args()
    
    if not args.video and not args.youtube:
        print("Please provide either a local video file or a YouTube URL")
        parser.print_help()
        return
    
    # Initialize the chapter generator
    generator = VideoChapterGenerator(model_size=args.model)
    
    # Process local video if provided
    if args.video:
        if not os.path.exists(args.video):
            print(f"Error: Video file not found: {args.video}")
            return
        process_video(generator, args.video)
    
    # Process YouTube video if provided
    if args.youtube:
        process_video(generator, args.youtube)

if __name__ == "__main__":
    main() 