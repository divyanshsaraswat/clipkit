import { loadTextModel, embedText } from '../../src/text.js';
(async () => {
  await loadTextModel();
  const emb = await embedText('a photo of a cat');
  console.log('Text embedding:', emb);
})();