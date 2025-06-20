import * as ort from "onnxruntime-node";
import { ensureModel } from "../src/model.js";
import sharp from "sharp";

let session: ort.InferenceSession | null = null;

/**
 * Loads the CLIP image ONNX model, downloading it if not already present.
 */
export async function loadImageModel(): Promise<void> {
  const modelPath = await ensureModel("image");
  session = await ort.InferenceSession.create(modelPath);
}

/**
 * Preprocess image into Float32Array in [1, 3, 224, 224] format, normalized for CLIP.
 */
async function preprocessImage(imagePath: string): Promise<Float32Array> {
  const imageBuffer = await sharp(imagePath)
    .resize(224, 224)
    .removeAlpha()
    .raw()
    .toBuffer();

  const floatArray = new Float32Array(3 * 224 * 224);
  const mean = [0.48145466, 0.4578275, 0.40821073];
  const std = [0.26862954, 0.26130258, 0.27577711];

  for (let i = 0; i < 224 * 224; i++) {
    for (let c = 0; c < 3; c++) {
      const val = imageBuffer[i * 3 + c] / 255.0;
      floatArray[c * 224 * 224 + i] = (val - mean[c]) / std[c];
    }
  }

  return floatArray;
}

/**
 * Embeds an image file using the CLIP model.
 * @param imagePath - Path to the image file.
 * @returns Embedding vector as a float array.
 */
export async function embedImage(imagePath: string): Promise<number[]> {
  if (!session) {
    throw new Error("Image model not loaded. Call loadImageModel() first.");
  }

  const imageData = await preprocessImage(imagePath);
  const inputName = session.inputNames[0];
  const tensor = new ort.Tensor("float32", imageData, [1, 3, 224, 224]);

  const feeds: Record<string, ort.Tensor> = {
    [inputName]: tensor,
  };

  const results = await session.run(feeds);
  const outputName = session.outputNames[0];
  const outputTensor = results[outputName];

  if (!outputTensor || !(outputTensor.data instanceof Float32Array)) {
    throw new Error(`Invalid or missing output from model for '${outputName}'`);
  }

  return Array.from(outputTensor.data);
}
