from libraries import math, PyPDF2, librosa, SeleniumURLLoader, pd

# Functions for file processing
def process_csv(file):
    df = pd.read_csv(file)
    return df.to_string()

def process_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text_data = ""
    for page in reader.pages:
        text_data += page.extract_text()
    return text_data

def process_text_file(file):
    return file.getvalue().decode("utf-8")

def process_url(url):
    loader = SeleniumURLLoader(urls=[url])
    data = loader.load()
    return data[0].page_content if data else None

def process_audio(file, processor, model):
    audio_array, sampling_rate = librosa.load(file, sr=16000)
    audio_array, _ = librosa.effects.trim(audio_array)
    chunk_size = 10 * 16000
    num_chunks = math.ceil(len(audio_array) / chunk_size)
    text_data = ""
    for i in range(num_chunks - 1):
        chunk = audio_array[i * chunk_size : min((i + 1) * chunk_size, len(audio_array))]

        input_features = processor(chunk, sampling_rate=sampling_rate, 
                                   return_tensors="pt").input_features
        predicted_ids = model.generate(input_features, max_length=10000,
                                       forced_decoder_ids=processor.get_decoder_prompt_ids
                                       (language="en", task="transcribe"))
        text_data += processor.batch_decode(predicted_ids, 
                                            skip_special_tokens=True)[0] + " "
    return text_data