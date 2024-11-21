import os
from pyannote.audio import Pipeline
from pydub import AudioSegment
from tqdm import tqdm
from pydub.exceptions import CouldntDecodeError
from configparser import ConfigParser
from pytube import YouTube
from zipfile import ZipFile
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

# Load configuration from config.ini
config = ConfigParser()
config.read("config.ini")

# Configuration
INPUT_DIRECTORY = config["DEFAULT"].get("input_directory", "data")
OUTPUT_DIRECTORY = config["DEFAULT"].get("output_directory", "output")
DIARIZATION_MODEL = config["DEFAULT"].get("diarization_model", "pyannote/speaker-diarization")
HF_TOKEN = config["DEFAULT"].get("hf_token", "")
GOOGLE_DRIVE_ENABLED = config["DEFAULT"].getboolean("google_drive_enabled", False)

# Function to convert unsupported files to WAV
def convert_to_wav(file_path):
    print(f"Converting {file_path} to WAV format...")
    try:
        audio = AudioSegment.from_file(file_path)
        wav_path = os.path.splitext(file_path)[0] + ".wav"
        audio.export(wav_path, format="wav")
        print(f"File converted to: {wav_path}")
        return wav_path
    except CouldntDecodeError:
        print(f"Error: Unable to decode file {file_path}. Skipping...")
        return None
    except Exception as e:
        print(f"Unexpected error while converting {file_path}: {e}")
        return None

# Function to perform speaker diarization
def diarize_audio(file_path, output_dir, pipeline):
    print(f"Processing file: {file_path}")
    
    try:
        # Load audio
        audio = AudioSegment.from_file(file_path)
    except CouldntDecodeError:
        print(f"Error: Could not decode file {file_path}. Skipping...")
        return
    except Exception as e:
        print(f"Unexpected error while loading {file_path}: {e}")
        return

    try:
        # Perform diarization
        diarization = pipeline(file_path)
    except Exception as e:
        print(f"Error during diarization of {file_path}: {e}")
        return
    
    # Prepare speaker-specific directories
    os.makedirs(output_dir, exist_ok=True)

    # Split and save segments
    total_segments = sum(1 for _ in diarization.itertracks(yield_label=True))
    with tqdm(total=total_segments, desc=f"Splitting audio for {os.path.basename(file_path)}") as pbar:
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            # Ensure directory for this speaker exists
            speaker_dir = os.path.join(output_dir, f"Speaker_{speaker}")
            os.makedirs(speaker_dir, exist_ok=True)
            
            # Extract and save the audio segment
            start_ms = int(segment.start * 1000)
            end_ms = int(segment.end * 1000)
            segment_audio = audio[start_ms:end_ms]
            
            # Save the segment
            segment_filename = os.path.join(speaker_dir, f"{os.path.basename(file_path)}_{start_ms}-{end_ms}.wav")
            segment_audio.export(segment_filename, format="wav")
            pbar.update(1)

    print(f"Finished processing file: {file_path}")

# Function to process files in a directory
def process_audio_files(input_dir, output_dir):
    print(f"Initializing diarization pipeline...")
    try:
        pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=HF_TOKEN)
    except Exception as e:
        print(f"Error initializing diarization pipeline: {e}")
        return
    
    print(f"Processing audio files in {input_dir}...")
    for root, _, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
                if not file.lower().endswith(".wav"):
                    # Convert non-WAV files to WAV
                    file_path = convert_to_wav(file_path)
                    if not file_path:  # Skip if conversion failed
                        continue
                diarize_audio(file_path, output_dir, pipeline)

# Function to upload to Google Drive
def upload_to_google_drive(local_folder, zip_name="processed_audio.zip"):
    print("Zipping processed files...")
    zip_path = os.path.join(local_folder, zip_name)
    with ZipFile(zip_path, "w") as zipf:
        for root, _, files in os.walk(local_folder):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, local_folder))
    print(f"Files zipped as {zip_path}")
    
    print("Uploading to Google Drive...")
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)
    drive_file = drive.CreateFile({"title": zip_name})
    drive_file.SetContentFile(zip_path)
    drive_file.Upload()
    print(f"Uploaded {zip_name} to Google Drive.")

# Main function
if __name__ == "__main__":
    process_audio_files(INPUT_DIRECTORY, OUTPUT_DIRECTORY)
    if GOOGLE_DRIVE_ENABLED:
        upload_to_google_drive(OUTPUT_DIRECTORY)
