import type { SceneJSON, Piece } from "./types";

// True rule for the demo task: "there are exactly 3 flat pieces".
// A piece counts as flat if:
//   - shape is block/pyramid and orientation is "flat", OR
//   - shape is wedge and orientation is "cheesecake" or "doorstop".
export function isFlat(p: Piece): boolean {
  if (p.orientation === "flat") return true;
  if (p.shape === "wedge" && (p.orientation === "cheesecake" || p.orientation === "doorstop")) {
    return true;
  }
  return false;
}

export function countFlat(scene: SceneJSON): number {
  return scene.pieces.reduce((acc, p) => acc + (isFlat(p) ? 1 : 0), 0);
}

export function trueLabel(scene: SceneJSON): "YES" | "NO" {
  return countFlat(scene) === 3 ? "YES" : "NO";
}

export const TRUE_RULE_TEXT = "There are exactly 3 flat pieces.";

const WORD_NUMBERS: Record<string, string> = {
  zero: "0", one: "1", two: "2", three: "3", four: "4",
  five: "5", six: "6", seven: "7", eight: "8", nine: "9",
};

function normalize(s: string): string {
  let out = s.toLowerCase().trim();
  // Replace punctuation with spaces so word boundaries stay clean.
  out = out.replace(/[.,;:!?"'()\-_/]+/g, " ");
  // Collapse whitespace
  out = out.replace(/\s+/g, " ");
  // Word numbers → digits (word boundaries only)
  out = out.replace(/\b(zero|one|two|three|four|five|six|seven|eight|nine)\b/g, (m) => WORD_NUMBERS[m]);
  return out.trim();
}

// Disqualifying phrases: if any of these appear, the guess is not the true rule
// even if it mentions "3" and "flat" (e.g. "at least 3 flat", "not exactly 3 flat").
const NEGATION_PATTERNS: RegExp[] = [
  /\bat least\b/,
  /\bat most\b/,
  /\bmore than\b/,
  /\bgreater than\b/,
  /\bfewer than\b/,
  /\bless than\b/,
  /\bnot\b/,
  /\bno\b/,
  /\bnever\b/,
  /\bexcept\b/,
  /\bwithout\b/,
];

// Words that count as "flat" in the guess. Wedges rendered flat are called
// cheesecake/doorstop in the DSL, but a natural-language guess "3 flat pieces"
// should still match those.
const FLAT_WORD = /\b(flat|flats|flatly|cheesecake|doorstop)\b/;

// Accept common phrasings of "there are exactly 3 flat pieces". Examples that
// return true:
//   "3 flat"
//   "3 flats"
//   "exactly 3 are flat"
//   "exactly three are flat"
//   "there are exactly 3 flat pieces"
//   "the number of flat pieces is 3"
//   "count of flat objects equals 3"
//   "three of the pieces lie flat"
// Examples that return false:
//   "at least 3 flat pieces"
//   "not 3 flat pieces"
//   "3 pieces and a flat floor" (proximity check catches this)
export function ruleMatches(guess: string): boolean {
  const g = normalize(guess);
  if (!g) return false;

  // Must reference "3" and some flat-word.
  if (!/\b3\b/.test(g)) return false;
  const flatMatch = g.match(FLAT_WORD);
  if (!flatMatch) return false;

  // Must not carry a disqualifying quantifier.
  for (const pat of NEGATION_PATTERNS) {
    if (pat.test(g)) return false;
  }

  // Proximity check: "3" and the flat-word should be within a small window so
  // sentences like "3 pyramids and flat ground" don't accidentally match.
  const tokens = g.split(" ");
  const threeIdx = tokens.findIndex((t) => t === "3");
  const flatIdx = tokens.findIndex((t) => FLAT_WORD.test(t));
  if (threeIdx === -1 || flatIdx === -1) return false;
  if (Math.abs(threeIdx - flatIdx) > 5) return false;

  return true;
}
