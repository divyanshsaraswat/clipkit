import { loadImageModel, embedImage } from '../../src/image.js';
(async () => {
  await loadImageModel();
  const dummy = new Float32Array(3 * 224 * 224).fill(0.5);
  const emb = await embedImage('./test/manual/dog.jpg');
  console.log('Image embedding:', emb);
})();