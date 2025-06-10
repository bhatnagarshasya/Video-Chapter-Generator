class VideoChapterGeneratorError(Exception):
    """Base exception for VideoChapterGenerator."""
    pass

class VideoProcessingError(VideoChapterGeneratorError):
    """Raised when there's an error processing the video file."""
    pass

class AudioExtractionError(VideoChapterGeneratorError):
    """Raised when there's an error extracting audio from video."""
    pass

class TranscriptionError(VideoChapterGeneratorError):
    """Raised when there's an error during speech transcription."""
    pass

class ExportError(VideoChapterGeneratorError):
    """Raised when there's an error exporting chapters."""
    pass 