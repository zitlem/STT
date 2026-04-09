/**
 * AudioWorklet Processor for Remote Mic
 * Captures audio samples from the browser microphone and sends them to the main thread
 * for transmission to the server via Socket.IO
 *
 * Audio Format: 16kHz, mono, 16-bit signed integer PCM
 */

class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.bufferSize = 1600; // 100ms at 16kHz
        this.buffer = new Float32Array(this.bufferSize);
        this.bufferIndex = 0;
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (!input || !input[0]) {
            return true;
        }

        const channelData = input[0]; // Mono channel

        for (let i = 0; i < channelData.length; i++) {
            this.buffer[this.bufferIndex++] = channelData[i];

            // When buffer is full, convert to Int16 and send
            if (this.bufferIndex >= this.bufferSize) {
                const int16Buffer = this.float32ToInt16(this.buffer);
                this.port.postMessage(int16Buffer.buffer, [int16Buffer.buffer]);

                // Reset buffer
                this.buffer = new Float32Array(this.bufferSize);
                this.bufferIndex = 0;
            }
        }

        return true;
    }

    /**
     * Convert Float32 audio samples to Int16 PCM
     * Float32 range: -1.0 to 1.0
     * Int16 range: -32768 to 32767
     */
    float32ToInt16(float32Array) {
        const int16Array = new Int16Array(float32Array.length);
        for (let i = 0; i < float32Array.length; i++) {
            // Clamp to -1.0 to 1.0 range
            const sample = Math.max(-1, Math.min(1, float32Array[i]));
            // Convert to Int16
            int16Array[i] = sample < 0 ? sample * 32768 : sample * 32767;
        }
        return int16Array;
    }
}

registerProcessor('audio-processor', AudioProcessor);
