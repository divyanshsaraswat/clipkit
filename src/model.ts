import fs from "fs";
import path from "path";
import followRedirects from "follow-redirects";
const https = followRedirects.https;

const MODEL_DIR = path.resolve("onnx");

const MODELS = {
  text: {
    name: "clip-text.onnx",
    url: "https://github.com/divyanshsaraswat/onnx-models/releases/download/latest/clip-text.onnx",
  },
  image: {
    name: "clip-image.onnx",
    url: "https://github.com/divyanshsaraswat/onnx-models/releases/download/latest/clip-image.onnx",
  },
};

export async function ensureModel(type: "text" | "image"): Promise<string> {
  const model = MODELS[type];
  const filePath = path.join(MODEL_DIR, model.name);

  if (!fs.existsSync(MODEL_DIR)) {
    fs.mkdirSync(MODEL_DIR);
  }

  if (!fs.existsSync(filePath)) {
    console.log(`Downloading ${model.name}...`);
    await downloadFile(model.url, filePath);
    console.log(`${model.name} downloaded.`);
  }

  return filePath;
}

function downloadFile(url: string, dest: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest);

    https
      .get(url, (response) => {
        if (response.statusCode !== 200) {
          return reject(
            new Error(`Failed to get '${url}' (${response.statusCode})`)
          );
        }

        response.pipe(file);
        file.on("finish", () => {
          file.close();
          resolve();
        });
      })
      .on("error", (err) => {
        fs.unlink(dest, () => reject(err));
      });
  });
}
