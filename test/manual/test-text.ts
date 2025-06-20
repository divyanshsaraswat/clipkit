import { loadTextModel, embedText } from '../../src/text.js';
(async () => {
  await loadTextModel('https://github.com/divyanshsaraswat/onnx-models/releases/download/latest/clip-text.onnx');
  const emb = await embedText('a photo of a cat');
  console.log('Text embedding:', emb);
})();