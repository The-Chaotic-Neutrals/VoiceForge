from dataclasses import dataclass

@dataclass
class TrainConfig:
    # --- Paths ---
    # Directory where setup.py downloaded the files
    model_dir: str = "./pretrained_models"
    
    # Path to your metadata CSV (Format: ID|RawText|NormText)
    csv_path: str = r"D:\\Github\\VoiceForge\\app\\datasets\\chatterbox\\Soft\metadata.csv"
    metadata_path: str = "./metadata.json"
    
    # Directory containing WAV files
    wav_dir: str = r"D:\\Github\\VoiceForge\\app\\datasets\\chatterbox\\Soft\wavs"
    
    preprocessed_dir = r"D:\\Github\\VoiceForge\\app\\models\\chatterbox_custom\\Soft\\preprocess"
    
    # Output directory for the finetuned model
    output_dir: str = r"D:\\Github\\VoiceForge\\app\\models\\chatterbox_custom\\Soft"
    
    is_inference = False
    inference_prompt_path: str = "./speaker_reference/2.wav"
    inference_test_text: str = "Test text for inference."

    ljspeech = True  # LJSpeech format dataset
    json_format = False
    preprocess = True
    
    is_turbo: bool = True

    # --- Vocabulary ---
    new_vocab_size: int = 52260 if True else 2454 

    # --- Hyperparameters ---
    batch_size: int = 4
    grad_accum: int = 8
    learning_rate: float = 5e-05
    num_epochs: int = 150
    
    save_steps: int = 500
    save_total_limit: int = 2
    dataloader_num_workers: int = 0  # Must be 0 on Windows to avoid multiprocessing issues

    # --- Constraints ---
    start_text_token = 255
    stop_text_token = 0
    max_text_len: int = 256
    max_speech_len: int = 850
    prompt_duration: float = 3.0
