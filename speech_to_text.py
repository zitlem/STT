import argparse
import io
import os
import sqlite3
import logging
import signal
import threading
import multiprocessing
from datetime import timedelta, datetime
import pytz
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import torch
import whisper
import speech_recognition as sr


# Set the time zone to "America/Los_Angeles" (Pacific Time Zone)
seattle_timezone = pytz.timezone('America/Los_Angeles')


# Generate the current date and time as a string
current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M_%A')
current_year = datetime.now().strftime('%Y')
current_month = datetime.now().strftime('%Y-%m')

# Create the folder if it doesn't exist
#folder_name = 'transcriptions'
folder_name =  f'_AUTOMATIC_BACKUP/{current_year}/{current_month}'
os.makedirs(folder_name, exist_ok=True)

# Create the database file path with the current date and time
db_name = os.path.join(folder_name, f'{current_datetime}_Transcriptions.db')

# Connect to the dynamically named database
db_connection = sqlite3.connect(db_name)
db_cursor = db_connection.cursor()

# Create the table if it doesn't exist
db_cursor.execute('''CREATE TABLE IF NOT EXISTS transcriptions (timestamp TEXT,text TEXT)''')
db_connection.commit()

# Insert a blank first entry with default values
default_timestamp = ' '  # You can use a default timestamp value here
default_text = ' '       # You can use a default text value here

db_cursor.execute('INSERT INTO transcriptions (timestamp, text) VALUES (?, ?)', (default_timestamp, default_text))
db_connection.commit()

db_connection.close()





app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_keyss'
socketio = SocketIO(app, static_url_path='/static', static_folder='static' )

app_logger = logging.getLogger(__name__)  # Use your module name here
socket_io_logger = logging.getLogger('socketio')

# Set log levels as needed
app_logger.setLevel(logging.DEBUG)
socket_io_logger.setLevel(logging.WARNING)

# Disable Flask's built-in logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/black')
def index2():
    return render_template('index2.html')


@app.route('/favicon.ico')
def favicon():
    # Return a custom response or an empty response
    return '', 204


@socketio.on('connect')
def handle_connect():
    #print('Client connected')
    emit('connected', {'data': 'Connected to Alexs server'})

@socketio.on('disconnect')
def handle_disconnect():
    emit('connected', {'data': 'Disconnected from Alexs server'})


def get_new_entries():
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM transcriptions ORDER BY timestamp DESC')
    transcriptions = cursor.fetchall()
    conn.close()
    return transcriptions

def emit_new_entries():
    while True:
        new_entries = get_new_entries()
        socketio.emit('new_entries', new_entries)
        socketio.sleep(.5)  # Emit updates every 5 seconds


def thread1_function():
    try:
        while True:
            def main():
                parser = argparse.ArgumentParser()
                parser.add_argument("--model", default="tiny", help="Model to use",
                                    choices=["tiny", "base", "small", "medium", "large","large-v1","large-v2"])
                parser.add_argument("--non_english", action='store_false',
                                    help="Don't use the english model.")
                parser.add_argument("--energy_threshold", default=3500,
                                    help="Energy level for mic to detect.", type=int)
                parser.add_argument("--record_timeout", default=3,
                                    help="How real time the recording is in seconds.", type=float)
                parser.add_argument("--phrase_timeout", default=2,
                                    help="How much empty space between recordings before we "
                                        "consider it a new line in the transcription.", type=float)  
                if 'linux' in platform:
                    parser.add_argument("--default_microphone", default='pulse',
                                        help="Default microphone name for SpeechRecognition. "
                                            "Run this with 'list' to view available Microphones.", type=str)
                args = parser.parse_args()
                
                # The last time a recording was retreived from the queue.
                phrase_time = None
                # Current raw audio bytes.
                last_sample = bytes()
                # Thread safe Queue for passing data from the threaded recording callback.
                data_queue = Queue()
                # We use SpeechRecognizer to record our audio because it has a nice feauture where it can detect when speech ends.
                recorder = sr.Recognizer()
                recorder.energy_threshold = args.energy_threshold
                # Definitely do this, dynamic energy compensation lowers the energy threshold dramtically to a point where the SpeechRecognizer never stops recording.
                recorder.dynamic_energy_threshold = False
                
                # Important for linux users. 
                # Prevents permanent application hang and crash by using the wrong Microphone
                if 'linux' in platform:
                    mic_name = args.default_microphone
                    if not mic_name or mic_name == 'list':
                        print("Available microphone devices are: ")
                        for index, name in enumerate(sr.Microphone.list_microphone_names()):
                            print(f"Microphone with name \"{name}\" found")   
                        return
                    else:
                        for index, name in enumerate(sr.Microphone.list_microphone_names()):
                            if mic_name in name:
                                source = sr.Microphone(sample_rate=16000, device_index=index)
                                break
                else:
                    source = sr.Microphone(sample_rate=16000)
                    
                # Load / Download model
                model = args.model
                if args.model != "large" and not args.non_english:
                    model = model + ".en"
                audio_model = whisper.load_model(model)

                record_timeout = args.record_timeout
                phrase_timeout = args.phrase_timeout

                temp_file = NamedTemporaryFile().name
                transcription = ['']
                
                with source:
                    recorder.adjust_for_ambient_noise(source)

                def record_callback(_, audio:sr.AudioData) -> None:
                    """
                    Threaded callback function to recieve audio data when recordings finish.
                    audio: An AudioData containing the recorded bytes.
                    """
                    # Grab the raw bytes and push it into the thread safe queue.
                    data = audio.get_raw_data()
                    data_queue.put(data)

                # Create a background thread that will pass us raw audio bytes.
                # We could do this manually but SpeechRecognizer provides a nice helper.
                recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

                # Cue the user that we're ready to go.
                print("Model loaded.\n")

                while True:
                    try:
                        
                        # Get the current time in the Seattle time zone
                        now_seattle = datetime.now(seattle_timezone)
                                               
                        # Pull raw recorded audio from the queue.
                        if not data_queue.empty():
                            phrase_complete = False
                            # If enough time has passed between recordings, consider the phrase complete.
                            # Clear the current working audio buffer to start over with the new data.
                            if phrase_time and now_seattle - phrase_time > timedelta(seconds=phrase_timeout):
                                last_sample = bytes()
                                phrase_complete = True
                            # This is the last time we received new audio data from the queue.
                            phrase_time = now_seattle

                            # Concatenate our current audio data with the latest audio data.
                            while not data_queue.empty():
                                data = data_queue.get()
                                last_sample += data

                            # Use AudioData to convert the raw data to wav data.
                            audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                            wav_data = io.BytesIO(audio_data.get_wav_data())

                            # Write wav data to the temporary file as bytes.
                            with open(temp_file, 'w+b') as f:
                                f.write(wav_data.read())

                            # Read the transcription.
                            result = audio_model.transcribe(temp_file, fp16=torch.cuda.is_available())
                            text = result['text'].strip()



                            # If we detected a pause between recordings, add a new item to our transcripion.
                            # Otherwise edit the existing one.
                            if phrase_complete:
                                transcription.append(text)
                            else:
                                transcription[-1] = text

                            # Insert the transcription into the database only if text is not empty                    
                            if text:
                                # Establish a new database connection and cursor for this thread
                                thread_db_connection = sqlite3.connect(db_name)
                                thread_db_cursor = thread_db_connection.cursor()

                               
                                timestamp = now_seattle.strftime('%Y-%m-%d %H:%M:%S')
                                
                                thread_db_cursor.execute("INSERT INTO transcriptions (timestamp, text) VALUES (?, ?)", (timestamp, text))
                                thread_db_connection.commit()

                                # Close the thread-specific database connection and cursor
                                thread_db_connection.close()

                            # Clear the console to reprint the updated transcription.
                            #os.system('cls' if os.name=='nt' else 'clear')
                            #for line in transcription:
                            #   print(line)          
                            
                            # Flush stdout.
                            #print('', end='', flush=True)

                            # Infinite loops are bad for processors, must sleep.
                            sleep(0.25)
                    except KeyboardInterrupt:
                        break
            main()
    except KeyboardInterrupt:
        print("Thread 1 received KeyboardInterrupt")
        os._exit(0)

def thread2_function():
    try:
        while True:
            socketio.start_background_task(emit_new_entries)
            app.run(host='0.0.0.0', port=80,debug=False) 
    except KeyboardInterrupt:
        print("Thread 2 received KeyboardInterrupt")
        os._exit(0)

def signal_handler(signum, frame):
    print("Interrupt signal received, stopping threads...")
    thread1.join()
    thread2.join()
    print("Threads stopped.")
    exit()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)

    thread1 = multiprocessing.Process(target=thread1_function)
    #thread2 = multiprocessing.Process(target=thread2_function)
    #thread1 = threading.Thread(target=thread1_function)
    thread2 = threading.Thread(target=thread2_function)

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()