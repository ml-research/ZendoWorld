// Pre-rendered scene images for the demo task ("exactly 3 flat pieces").
// Positive examples satisfy the rule; negative examples (filenames with `_n`)
// do not. Vite hashes these paths and copies them into the build output.

import pos1 from "./assets/scenes/47_1.png";
import pos2 from "./assets/scenes/47_2.png";
import pos3 from "./assets/scenes/47_9.png";
import neg1 from "./assets/scenes/47_3_n.png";
import neg2 from "./assets/scenes/47_8_n.png";
import neg3 from "./assets/scenes/47_9_n.png";

export const POSITIVE_SCENES: string[] = [pos1, pos2, pos3];
export const NEGATIVE_SCENES: string[] = [neg1, neg2, neg3];
