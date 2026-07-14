import type { Shape, ColorName, Orientation, Piece } from "./types";

// BLOCK
import block_upright_red from "./assets/item_red_block_upright.png";
import block_upside_down_red from "./assets/item_red_block_upside_down.png";
import block_flat_red from "./assets/item_red_block_flat.png";
import block_upright_yellow from "./assets/item_yellow_block_upright.png";
import block_upside_down_yellow from "./assets/item_yellow_block_upside_down.png";
import block_flat_yellow from "./assets/item_yellow_block_flat.png";
import block_upright_blue from "./assets/item_blue_block_upright.png";
import block_upside_down_blue from "./assets/item_blue_block_upside_down.png";
import block_flat_blue from "./assets/item_blue_block_flat.png";

// WEDGE (Achtung: kein `flat`; und deine zwei Pfade waren vertauscht)
import wedge_cheesecake_yellow from "./assets/item_yellow_wedge_cheesecake.png";
import wedge_doorstop_yellow from "./assets/item_yellow_wedge_doorstop.png";
import wedge_upright_yellow from "./assets/item_yellow_wedge_upright.png";
import wedge_upside_down_yellow from "./assets/item_yellow_wedge_upside_down.png";

import wedge_cheesecake_red from "./assets/item_red_wedge_cheesecake.png";
import wedge_doorstop_red from "./assets/item_red_wedge_doorstop.png";
import wedge_upright_red from "./assets/item_red_wedge_upright.png";
import wedge_upside_down_red from "./assets/item_red_wedge_upside_down.png"; // <-- war zuvor blau

import wedge_cheesecake_blue from "./assets/item_blue_wedge_cheesecake.png";
import wedge_doorstop_blue from "./assets/item_blue_wedge_doorstop.png";
import wedge_upright_blue from "./assets/item_blue_wedge_upright.png";
import wedge_upside_down_blue from "./assets/item_blue_wedge_upside_down.png"; // <-- war zuvor gelb
// import wedge_flat_blue ...  <-- ENTFALLEN (gibt es nicht)

// PYRAMID
import pyramid_upright_blue from "./assets/item_blue_pyramid_upright.png";
import pyramid_upside_down_blue from "./assets/item_blue_pyramid_upside_down.png";
import pyramid_flat_blue from "./assets/item_blue_pyramid_flat.png";
import pyramid_upright_yellow from "./assets/item_yellow_pyramid_upright.png";
import pyramid_upside_down_yellow from "./assets/item_yellow_pyramid_upside_down.png";
import pyramid_flat_yellow from "./assets/item_yellow_pyramid_flat.png";
import pyramid_upright_red from "./assets/item_red_pyramid_upright.png";
import pyramid_upside_down_red from "./assets/item_red_pyramid_upside_down.png";
import pyramid_flat_red from "./assets/item_red_pyramid_flat.png";

type Key = `${Shape}_${Orientation}_${ColorName}`;

export const SPRITES: Partial<Record<Key, string>> = {
  // block
  block_upright_red,
  block_upside_down_red,
  block_flat_red,
  block_upright_yellow,
  block_upside_down_yellow,
  block_flat_yellow,
  block_upright_blue,
  block_upside_down_blue,
  block_flat_blue,
  // wedge
  wedge_cheesecake_yellow,
  wedge_doorstop_yellow,
  wedge_upright_yellow,
  wedge_upside_down_yellow,
  wedge_cheesecake_red,
  wedge_doorstop_red,
  wedge_upright_red,
  wedge_upside_down_red,
  wedge_cheesecake_blue,
  wedge_doorstop_blue,
  wedge_upright_blue,
  wedge_upside_down_blue,
  // pyramid
  pyramid_upright_blue,
  pyramid_upside_down_blue,
  pyramid_flat_blue,
  pyramid_upright_yellow,
  pyramid_upside_down_yellow,
  pyramid_flat_yellow,
  pyramid_upright_red,
  pyramid_upside_down_red,
  pyramid_flat_red,
};

export function spriteFor(p: Piece): string | undefined {
  const key = `${p.shape}_${p.orientation}_${p.color}` as Key;
  const src = SPRITES[key];
  // statt process.env:
  if (!src && import.meta.env?.DEV) {
    console.warn("[sprites] Missing sprite for", key);
  }
  return src;
}
