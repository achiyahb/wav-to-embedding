import torch
import multiprocessing
from flask import Flask, request, jsonify
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import io
import soundfile as sf

if __name__ == '__main__':
    multiprocessing.set_start_method('fork')

    app = Flask(__name__)

    # Load processor and model
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base", cache_dir="./model_cache")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", cache_dir="./model_cache")

    @app.route('/health', methods=['POST'])
    def health_check():
        return 'healthy'

    @app.route('/extract', methods=['POST'])
    def extract_embedding():
        file = request.files['file']
        audio_bytes = io.BytesIO(file.read())
        audio_bytes.seek(0)
        
        # Read audio
        waveform, sample_rate = sf.read(audio_bytes, dtype="float32")
        print("Shape after sf.read:", waveform.shape)  # e.g. (samples, channels) or (samples,)

        # Downmix to mono if multi-channel
        if waveform.ndim == 2:
            # waveform shape is (samples, channels)
            waveform = waveform.mean(axis=1)  # (samples,)
        print("Shape after downmix (if applied):", waveform.shape)

        # Convert to torch tensor
        waveform = torch.tensor(waveform)  # (samples,)
        print("After torch.tensor:", waveform.shape)

        # Resample if needed
        if sample_rate != 16000:
            print(f"Resampling from {sample_rate} to 16000")
            # Resample expects (batch, time). Add a batch dimension first:
            waveform = waveform.unsqueeze(0)  # (1, samples)
            print("Before resample:", waveform.shape)
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)  # (1, new_samples)
            print("After resample:", waveform.shape)
            # Remove the batch dimension after resampling to return to a simple 1D waveform
            waveform = waveform.squeeze(0)  # (new_samples,)
        
        print("Before processor, waveform shape:", waveform.shape)  # Expect (samples,)

        # Convert back to numpy for processor
        waveform = waveform.numpy()  # (samples,)
        print("Converted to numpy:", waveform.shape)

        # Pass as a list to the processor to handle batch dimension automatically
        inputs = processor([waveform], sampling_rate=16000, return_tensors="pt", padding=True)
        print("Processor input_values shape:", inputs["input_values"].shape)  # Expect (1, length)

        # Run the model
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

        return jsonify({"embedding": embedding})

    app.run(debug=True)
