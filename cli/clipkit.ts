#!/usr/bin/env node

import { program } from "commander";
import { loadTextModel, embedText, loadImageModel, embedImage } from "../src/index.js";
import path from "path";

/**
 * CLI to embed text or images using CLIP
 */

program
  .name("clipkit")
  .description("CLI to embed text and images using CLIP")
  .version("1.0.0");

// Text command
program
  .command("text <text>")
  .description("Embed text using CLIP")
  .action(async (text: string) => {
    try {
      await loadTextModel(); // assumes internal model download via `ensureModel`
      const embedding = await embedText(text);
      console.log(JSON.stringify(embedding));
    } catch (err) {
      console.error("Failed to embed text:", err);
    }
  });

// Image command
program
  .command("image <imagePath>")
  .description("Embed image using CLIP")
  .action(async (imagePath: string) => {
    try {
      await loadImageModel(); // assumes internal model download via `ensureModel`
      const embedding = await embedImage(path.resolve(imagePath));
      console.log(JSON.stringify(embedding));
    } catch (err) {
      console.error("Failed to embed image:", err);
    }
  });

program.parse(process.argv);
