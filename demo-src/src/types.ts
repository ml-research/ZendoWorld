// types.ts
export type Label = "YES" | "NO";
export type Shape = "block" | "pyramid" | "wedge";
export type ColorName = "red" | "blue" | "yellow";
export type BlockOrPyramidOrientation = "upright" | "upside_down" | "flat";
export type WedgeOrientation =
  | "upright"
  | "upside_down"
  | "cheesecake"
  | "doorstop";
export type Orientation = BlockOrPyramidOrientation | WedgeOrientation;

export type Piece = {
  id: string;
  shape: Shape;
  color: ColorName;
  x: number;
  y: number;
  z: number;
  orientation: Orientation;
  touchingLeft: string | null;
  touchingRight: string | null;
  onTop: string | null;
  below: string | null;
  pointing: string | null;
};

export type SceneJSON = {
  id: string;
  size: number;
  pieces: Piece[];
};

export type WSInitialExample = {
  imageDataUrl: string;
  label: Label;
};

export type WSMessage =
| { type: "system"; text: string; hypothesis?: string; stones?: number; sessionId?: string }
  | {
      type: "model_label";
      label: Label;
      hypothesis: string;
      stones: number;
    }
  | {
      type: "labeled_example";
      label: Label;
      imageDataUrl?: string | null;
      description?: string | null;
      isCounterExample?: boolean;
    }
  | { type: "guess"; guess: string; correct: boolean; stones: number }
  | {
      type: "quiz_result";
      correct: boolean;
      stones: number;
      playerId?: number;
    }
  | {
      type: "guess_rule_prompt";
      playerId?: number;
    }
  | {
      type: "rule_incorrect";
      rule: string;
    }
  | {
      type: "human_guess_label_request";
      imageDataUrl?: string | null;
      playerId?: number;
    }
  | {
      type: "human_propose_request";
      playerId?: number;
      examplesCount: number;
    }
  | {
      type: "human_scene_preview";
      playerId?: number;
      scene: SceneJSON;
      imageDataUrl?: string | null;
    }
  | {
      type: "update_other_player_stones";
      playerId: number;
      stones: number;
    }
  | { type: "game_system_message"; text: string }
  | { type: "player_finished"; text: string; exhausted?: boolean }
  | { type: "ping" };
