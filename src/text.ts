import * as ort from 'onnxruntime-node';
import { ensureModel } from '../src/model.js';
import { tokenize } from '../src/utils.js';

let session: ort.InferenceSession | null = null;

/**
 * Loads the CLIP text ONNX model, downloading it if not already present.
 */
export async function loadTextModel(): Promise<void> {
  const modelPath = await ensureModel("text");

  try {
    session = await ort.InferenceSession.create(modelPath);
    console.log("Text model loaded successfully.");
  } catch (error) {
    console.error("Failed to load text model:", error);
    throw error;
  }
}

/**
 * Generate embeddings for a given text string.
 * @param text - Input string to embed.
 * @returns Embedding vector as a float array.
 */
export async function embedText(text: string): Promise<number[]> {
  if (!session) {
    throw new Error("Text model is not loaded. Call loadTextModel() first.");
  }

  const { inputIds, attentionMask } = tokenize(text);

  const inputs: Record<string, ort.Tensor> = {
    input_ids: new ort.Tensor('int64', BigInt64Array.from(inputIds.map(BigInt)), [1, inputIds.length]),
    attention_mask: new ort.Tensor('int64', BigInt64Array.from(attentionMask.map(BigInt)), [1, attentionMask.length]),
  };

  try {
    const output = await session.run(inputs);
    const outputName = session.outputNames[0];
    const result = output[outputName];

    if (!result || !(result.data instanceof Float32Array)) {
      throw new Error("Invalid output from text model.");
    }

    return Array.from(result.data);
  } catch (err) {
    console.error("Error during text embedding:", err);
    throw err;
  }
}
