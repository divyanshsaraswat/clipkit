# clipkit-embeddings

[![npm version](https://img.shields.io/npm/v/clipkit.svg)](https://www.npmjs.com/package/clipkit)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Issues](https://img.shields.io/github/issues/YOUR_USERNAME/clipkit)](https://github.com/YOUR_USERNAME/clipkit/issues)

> Fast CLIP text and image embedding toolkit for Node.js and browser using ONNX.

**clipkit-embeddings** is a TypeScript library and CLI that enables high-performance [CLIP](https://openai.com/research/clip) embeddings for both **text** and **images**, powered by `onnxruntime-node`. It allows you to use CLIP models offline for semantic search, similarity computation, or zero-shot classification — both programmatically and via the command line.

---

## ✨ Features

- 🔤 Embed text using `clip-text.onnx`
- 🖼️ Embed images using `clip-image.onnx`
- 🧠 Cosine similarity for comparing embeddings
- ⚡️ Fast inference with ONNXRuntime
- 🧰 CLI tool for quick use in scripts
- 🌐 Works in Node.js and browser (with bundler)

---

## 📦 Installation

### As a library:

```bash
npm install clipkit-embeddings
```
### As a library:
```bash
npm install -g clipkit-embeddings
```

## 🚀 Usage

### 🧠 Text Embedding

```ts
import { loadTextModel, embedText } from "clipkit-embeddings";

await loadTextModel("onnx/clip-text.onnx");
const embedding = await embedText("a photo of a dog");
console.log(embedding);
```

---

### 🖼️ Image Embedding

```ts
import { loadImageModel, embedImage } from "clipkit-embeddings";
import { preprocessImage } from "clipkit-embeddings/utils";

await loadImageModel("onnx/clip-image.onnx");
const tensor = await preprocessImage("test/cat.jpg");
const embedding = await embedImage(tensor);
console.log(embedding);
```

---

### 🔁 Cosine Similarity

```ts
import { cosineSimilarity } from "clipkit-embeddings/utils";

const similarity = cosineSimilarity(imageEmbedding, textEmbedding);
console.log("Cosine similarity:", similarity.toFixed(4));
```

---

## 🧪 CLI Usage

Embed text:

```bash
clipkit text "a photo of a cat"
```

Embed image:

```bash
clipkit image ./path/to/image.jpg
```

---

## 📁 Project Structure

```bash
clipkit-embeddings/
├── cli/              # CLI logic
├── src/              # Core logic (text/image/utils)
├── test/             # Sample test files and images
├── onnx/             # Model files (.onnx)
│   ├── clip-text.onnx
│   └── clip-image.onnx
├── package.json
├── tsconfig.json
├── README.md
└── LICENSE
```

---


## 👨‍💻 Contributing

Contributions, issues and feature requests are welcome!  
Please open an issue to discuss any major changes beforehand.

---

## 📚 References

- [CLIP Paper (OpenAI)](https://openai.com/research/clip)
- [ONNX Runtime](https://onnxruntime.ai/)
- [onnxruntime-node on npm](https://www.npmjs.com/package/onnxruntime-node)
- [Sharp image processing](https://www.npmjs.com/package/sharp)

---

## 📝 License

This project is licensed under the **MIT License**.  
See the [LICENSE](./LICENSE) file for full text.

---

## ✍️ Author

**Divyansh Saraswat**  
[GitHub](https://github.com/divyanshsaraswat) • [LinkedIn](https://linkedin.com/in/imdivyanshmv) • [Twitter](https://twitter.com/imdivyanshmv)