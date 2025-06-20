import { embedImage, loadImageModel } from "../../src/image.js";
import { embedText, loadTextModel } from "../../src/text.js";

/**
 * Compute cosine similarity between two vectors.
 */
function cosineSimilarity(a: number[], b: number[]): number {
  const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const normA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const normB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  return dot / (normA * normB);
}

async function run() {
  // Load ONNX models
  await loadTextModel();
  await loadImageModel();

  // Image and text input
  const imagePath = "./test/manual/dog.jpg"; // Change to your image path
  const textPrompt = "a dog";

  // Get embeddings
  const imageEmbedding = await embedImage(imagePath);
  const textEmbedding = await embedText(textPrompt);

  // Compute cosine similarity
  const similarity = cosineSimilarity(imageEmbedding, textEmbedding);
  console.log(`Cosine similarity: ${similarity.toFixed(4)}`);
}

run().catch(console.error);
