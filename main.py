import torch
import multiprocessing
from flask import Flask, request, jsonify
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import io

if __name__ == '__main__':
    multiprocessing.set_start_method('fork')

    app = Flask(__name__)

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base", cache_dir="./model_cache")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", cache_dir="./model_cache")

    @app.route('/health', methods=['POST'])
    def health_check():
        return 'healthy'

    @app.route('/extract', methods=['POST'])
    def extract_embedding():
        
        file = request.files['file']
        audio_bytes = io.BytesIO(file.read())  # ממיר את הקובץ ל-BytesIO
        waveform, sample_rate = torchaudio.load(audio_bytes, format="wav")
      
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        inputs = processor(waveform.squeeze(0), sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        return jsonify({"embedding": embedding})

    app.run(debug=True)
