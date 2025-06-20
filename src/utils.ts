export function tokenize(text: string) {
  const words = text.trim().toLowerCase().split(/\s+/);
  const vocab: Record<string, number> = {};
  words.forEach((w, i) => (vocab[w] = i + 1));
  const inputIds = words.map(w => vocab[w] || 0);
  const attentionMask = inputIds.map(() => 1);
  return { inputIds, attentionMask };
}

export function cosineSimilarity(a: number[], b: number[]): number {
  const dot = a.reduce((acc, val, i) => acc + val * b[i], 0);
  const magA = Math.sqrt(a.reduce((acc, val) => acc + val * val, 0));
  const magB = Math.sqrt(b.reduce((acc, val) => acc + val * val, 0));
  return dot / (magA * magB);
}