import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from faster_whisper import WhisperModel

# Configuration variables
INPUT_FOLDER = "data\\example\\wavs"
OUTPUT_FILE = "data\\example\\metadata.list"
LANGUAGE_CODE = "EN"
LANGUAGE_MODEL = "EN-default"

# Initialize the faster-whisper model
model = WhisperModel("medium", device="cuda", compute_type="int8_float16")

# Get the list of WAV files in the input directory
wav_files = [file for file in os.listdir(INPUT_FOLDER) if file.endswith(".wav")]

# Sort the WAV files in numeric order
wav_files = sorted(wav_files, key=lambda x: int(os.path.splitext(x)[0]))

# Open a text file for writing the transcripts
with open(OUTPUT_FILE, "w", encoding="utf-8") as transcript_file:
    # Prepare the output path with forward slashes
    output_path = INPUT_FOLDER.replace('\\', '/')
    
    # Iterate through each WAV file
    for i, wav_file in enumerate(wav_files):
        print(f"Transcribing: {wav_file}")
        # Construct the full path to the WAV file
        wav_path = os.path.join(INPUT_FOLDER, wav_file)
        
        # Transcribe using faster-whisper
        segments, info = model.transcribe(wav_path)
        # Get the full text from all segments
        transcribed_text = " ".join([segment.text for segment in segments]).strip()
        
        # Write the result to the transcript file in the MeloTTS format
        line = f"{output_path}/{wav_file}|{LANGUAGE_MODEL}|{LANGUAGE_CODE}|{transcribed_text}"
        if i < len(wav_files) - 1:
            line += "\n"
        transcript_file.write(line)

print(f"Transcription complete. Check '{OUTPUT_FILE}' for results.")