//哈夫曼编码
const RedAllCandidates = new Array(90).fill(0).map((a, i) => 89 - i);
const BlackAllCandidates = new Array(90).fill(0).map((a, i) => i);
const RedSoliderCandidates = new Array(45).fill(0).map((a, i) => 44 - i);
const BlackSoliderCandidates = new Array(45).fill(0).map((a, i) => 45 + i);
/*
黑
0 1 2 ... 8
9 10 11 ... 17
.   .  .      .
.   .  .      .
.   .  .      .
81  82 83 ... 89 
红
**/

//32位数组
export const PieceCandidates = [
  [85, 86, 84, 76, 77, 75, 67, 68, 66, 90],//红帅
  [90, 86, 84, 76, 68, 66],//红士1-右
  [90, 84, 86, 76, 66, 68],//红士2-左
  [90, 87, 67, 71, 51, 83, 47, 63],//红相1-右
  [90, 83, 67, 63, 47, 87, 51, 71],//红相2-左
  [90, ...RedAllCandidates],//红车1 0-89
  [90, ...RedAllCandidates],//红车2 0-89
  [90, ...RedAllCandidates],//红马1 0-89
  [90, ...RedAllCandidates],//红马2 0-89
  [90, ...RedAllCandidates],//红炮1 0-89
  [90, ...RedAllCandidates],//红炮2 0-89
  [90, 62, 53, ...RedSoliderCandidates],//红兵5 -最右
  [90, 60, 51, ...RedSoliderCandidates],//红兵4
  [90, 58, 49, ...RedSoliderCandidates],//红兵3
  [90, 56, 47, ...RedSoliderCandidates],//红兵2
  [90, 54, 45, ...RedSoliderCandidates],//红兵1 -最左
  [4, 3, 5, 13, 12, 14, 22, 21, 23, 90],//黑将
  [90, 3, 5, 13, 21, 23],//黑仕1-左
  [90, 5, 3, 13, 23, 21],//黑仕2-右
  [90, 2, 22, 18, 38, 6, 42, 26],//黑象1-左
  [90, 6, 22, 26, 42, 2, 38, 18],//黑象2-右
  [90, ...BlackAllCandidates],//黑车1 0-89
  [90, ...BlackAllCandidates],//黑车2 0-89
  [90, ...BlackAllCandidates],//黑马1 0-89
  [90, ...BlackAllCandidates],//黑马2 0-89
  [90, ...BlackAllCandidates],//黑炮1 0-89
  [90, ...BlackAllCandidates],//黑炮2 0-89
  [90, 27, 36, ...BlackSoliderCandidates],//黑卒5 -最左
  [90, 29, 38, ...BlackSoliderCandidates],//黑卒4
  [90, 31, 40, ...BlackSoliderCandidates],//黑卒3
  [90, 33, 42, ...BlackSoliderCandidates],//黑卒2
  [90, 35, 44, ...BlackSoliderCandidates],//黑卒1 -最右
];
export const initMap = [
  85, 86, 84, 87, 83,
  81, 89, 82, 88, 64, 70,
  62, 60, 58, 56, 54,
  4, 3, 5, 2, 6,
  0, 8, 1, 7, 19, 25,
  27, 29, 31, 33, 35
];
const validarr = new Array(90).fill(0);
export const validMap = (arr) => {
  validarr.fill(0);

  for (let item of arr) {
    if (item === 90) continue;
    if (validarr[item] === 1) {
      return false; // 重复位置
    }
    validarr[item] = 1;
  }

  return true;
};

const ceilLog2Map = new Map([
  [1, 0],
  [2, 1],
  [3, 2],
  [4, 2],
  [6, 3],
  [8, 3],
  [10, 4],
  [17, 5],
  [48, 6],
  [91, 7],
]);

const floorLog2Map = new Map([
  [1, 0],
  [2, 1],
  [3, 1],
  [4, 2],
  [6, 2],
  [8, 3],
  [10, 3],
  [17, 4],
  [48, 5],
  [91, 6],
]);
//current: number offset:number bits:number bitsLength:number
function concatBits(current, offset, bits, bitsLength) {
  let newCurrent = current;
  let newOffset = offset;
  const newUint8 = [];
  if (offset + bitsLength < 8) {
    newCurrent |= bits << (8 - bitsLength - offset);
    newOffset += bitsLength;
  } else if (offset + bitsLength === 8) {
    newUint8.push(current | bits);
    newCurrent = 0;
    newOffset = 0;
  } else {
    newCurrent |= bits >> (offset - 8 + bitsLength);
    newUint8.push(newCurrent);
    newCurrent = (bits << (16 - offset - bitsLength)) & 0xff;
    newOffset = offset - 8 + bitsLength;
  }
  return [newCurrent, newOffset, newUint8];
}
//current: number offset:number candidateIndex:number candidateLength:number
function concatFlexibleBits(current, offset, candidateIndex, candidateLength) {
  const floorLog = floorLog2Map.get(candidateLength);
  const ceilLog = ceilLog2Map.get(candidateLength);
  const last = 2 ** floorLog;
  const beyond = candidateLength - last;
  if (floorLog === ceilLog || candidateIndex < last - beyond) {
    return concatBits(current, offset, candidateIndex, floorLog);
  }
  let newCurrent = current;
  let newOffset = offset;
  const array = [];
  let newUint8;
  if (candidateIndex < last) {
    [newCurrent, newOffset, newUint8] = concatBits(newCurrent, newOffset, candidateIndex, floorLog);
    array.push(...newUint8);
    [newCurrent, newOffset, newUint8] = concatBits(newCurrent, newOffset, 0, 1);
    array.push(...newUint8);
  } else {
    [newCurrent, newOffset, newUint8] = concatBits(newCurrent, newOffset, candidateIndex - beyond, floorLog);
    array.push(...newUint8);
    [newCurrent, newOffset, newUint8] = concatBits(newCurrent, newOffset, 1, 1);
    array.push(...newUint8);
  }
  return [newCurrent, newOffset, array];
}
//array:Uint8Array bitsOffset:number bitsLength:number
function readBits(array, bitsOffset, bitsLength) {
  const offset = bitsOffset % 8;
  const index = Math.floor(bitsOffset / 8);
  if ((offset + bitsLength > 8 && index + 1 >= array.length) || offset + bitsLength <= 8 && index >= array.length) {
    throw new Error('readBitsError');
  }
  let number = offset + bitsLength <= 8 ? array[index] : (array[index] << 8) | array[index + 1];
  const length = offset + bitsLength <= 8 ? 8 : 16;
  number >>= (length - bitsLength - offset);
  number &= ([0, 1, 3, 7, 15, 31, 63][bitsLength]);
  return [number, bitsOffset + bitsLength];
}
//array:Uint8Array bitsOffset:number candidateLength:number
function readFlexibleBits(array, bitsOffset, candidateLength) {
  const floorLog = floorLog2Map.get(candidateLength);
  const ceilLog = ceilLog2Map.get(candidateLength);
  const last = 2 ** floorLog;
  const beyond = candidateLength - last;
  const [number, offset] = readBits(array, bitsOffset, floorLog);
  if (floorLog === ceilLog || number < last - beyond) {
    return [number, offset];
  }
  const [current, offset2] = readBits(array, offset, 1);
  if (current) {
    return [number + beyond, offset2];
  }
  return [number, offset2];
}
export function encoding(initMap) {
  let current = 0;
  let offset = 0;
  const output = [];

  for (let i = 0; i < initMap.length; i++) {
    const pos = initMap[i];
    const candidates = PieceCandidates[i];

    const candidateIndex = candidates.indexOf(pos);
    if (candidateIndex === -1) {
      throw new Error(`Invalid position for piece ${i}: ${pos}`);
    }

    const [newCurrent, newOffset, arr] =
      concatFlexibleBits(
        current,
        offset,
        candidateIndex,
        candidates.length
      );

    current = newCurrent;
    offset = newOffset;

    output.push(...arr);
  }

  // 收尾：如果还有未flush的bit
  if (offset > 0) {
    output.push(current);
  }

  return new Uint8Array(output);
}
export function decoding(uint8) {
  let bitsOffset = 0;
  const result = [];

  for (let i = 0; i < PieceCandidates.length; i++) {
    const candidates = PieceCandidates[i];

    const [candidateIndex, nextOffset] =
      readFlexibleBits(uint8, bitsOffset, candidates.length);

    bitsOffset = nextOffset;

    const pos = candidates[candidateIndex];

    result.push(pos);
  }

  return result;
}

const encoded = encoding(initMap);
/*
Uint8Array(19) [
   10, 73, 32, 144, 19, 74,  4,
   33,  8, 66, 146,  9, 33, 16,
  163, 65,  8,  66, 16
]
*/
//console.log(encoded);
const base64_code = btoa(String.fromCharCode(...encoded));
//console.log(base64_code);
//CkkgkBNKBCEIQpIJIRCjQQhCEA==
const decoded = decoding(encoded);
/*
[
  85, 86, 84, 87, 83, 81, 89, 82, 88,
  64, 70, 62, 60, 58, 56, 54,  4,  3,
   5,  2,  6,  0,  8,  1,  7, 19, 25,
  27, 29, 31, 33, 35
]
*/
//console.log(decoded);


export function rc2num(row, col) {
  return row * 9 + col;
}
export function num2rc(num) {
  return [Math.floor(num / 9), num % 9];
}
export function piece2id(flag, piece, num) {
  // 红方
  if (flag === 'red') {
    switch (piece) {
      case '帅':
        return 0;

      case '士':
        // 红士：1-右(1), 2-左(2)
        return num === 1 ? 1 : 2;

      case '相':
        // 红相：1-右, 2-左
        return num === 1 ? 3 : 4;

      case '车':
        return num === 1 ? 5 : 6;

      case '马':
        return num === 1 ? 7 : 8;

      case '炮':
        return num === 1 ? 9 : 10;

      case '兵':
        // 红兵：5(最右) -> 11 ... 1(最左) -> 15
        return 11 + (5 - num);

      default:
        return -1;
    }
  }

  // 黑方
  if (flag === 'black') {
    switch (piece) {
      case '将':
        return 16;

      case '仕':
        // 黑仕：1左=17, 2右=18
        return num === 1 ? 17 : 18;

      case '象':
        // 黑象：1左=19, 2右=20
        return num === 1 ? 19 : 20;

      case '车':
        return num === 1 ? 21 : 22;

      case '马':
        return num === 1 ? 23 : 24;

      case '炮':
        return num === 1 ? 25 : 26;

      case '卒':
        // 黑卒：5最左=27 ... 1最右=31
        return 27 + (5 - num);

      default:
        return -1;
    }
  }

  return -1;
}
export function id2piece(id) {
  // 红方
  if (id === 0) return ['red', '帅', 0];

  if (id === 1) return ['red', '士', 1];
  if (id === 2) return ['red', '士', 2];

  if (id === 3) return ['red', '相', 1];
  if (id === 4) return ['red', '相', 2];

  if (id === 5) return ['red', '车', 1];
  if (id === 6) return ['red', '车', 2];

  if (id === 7) return ['red', '马', 1];
  if (id === 8) return ['red', '马', 2];

  if (id === 9) return ['red', '炮', 1];
  if (id === 10) return ['red', '炮', 2];

  // 红兵 11~15
  if (id >= 11 && id <= 15) {
    return ['red', '兵', 5 - (id - 11)];
  }

  // 黑方
  if (id === 16) return ['black', '将', 0];

  if (id === 17) return ['black', '仕', 1];
  if (id === 18) return ['black', '仕', 2];

  if (id === 19) return ['black', '象', 1];
  if (id === 20) return ['black', '象', 2];

  if (id === 21) return ['black', '车', 1];
  if (id === 22) return ['black', '车', 2];

  if (id === 23) return ['black', '马', 1];
  if (id === 24) return ['black', '马', 2];

  if (id === 25) return ['black', '炮', 1];
  if (id === 26) return ['black', '炮', 2];

  // 黑卒 27~31
  if (id >= 27 && id <= 31) {
    return ['black', '卒', 5 - (id - 27)];
  }

  return null;
}
export function encode_action(action_list) {
  //action_list:[piece_id,action_id]
  let actions = piece2actions[id2piece(action_list[0])[1]];
  let offset = actions.indexOf(action_list[1]);
  let basei = action_map3[action_list[0]];
  return basei + offset;
}
//严格大于
function upper_bound(arr, target) {
  let l = 0, r = arr.length;

  while (l < r) {
    let m = (l + r) >> 1;
    if (arr[m] <= target) l = m + 1;
    else r = m;
  }

  return l;
}
export function decode_action(action) {
  // console.log("action_map3:", action_map3);
  // console.log("target_decode:", action);
  let piece_id = upper_bound(action_map3, action) - 1;//第一个大于action的索引
  //console.log("piece_id:", piece_id);
  let offset = action - action_map3[piece_id];
  let offset_action = piece2actions[id2piece(piece_id)[1]][offset];
  return [piece_id, offset_action];
}
const piece2actions = {
  "帅": [0, 1, -1, 9, -9],
  "将": [0, 1, -1, 9, -9],
  "士": [8, 10, -8, -10],
  "仕": [8, 10, -8, -10],
  "相": [20, 16, -16, -20],
  "象": [20, 16, -16, -20],
  "兵": [1, -1, -9],
  "卒": [1, -1, 9],
  "马": [17, 19, 11, 7, -17, -19, -11, -7],
  "车": [1, 2, 3, 4, 5, 6, 7, 8, 9, 18, 27, 36, 45, 54, 63, 72, 81, -1, -2, -3, -4, -5, -6, -7, -8, -9, -18, -27, -36, -45, -54, -63, -72, -81],
  "炮": [1, 2, 3, 4, 5, 6, 7, 8, 9, 18, 27, 36, 45, 54, 63, 72, 81, -1, -2, -3, -4, -5, -6, -7, -8, -9, -18, -27, -36, -45, -54, -63, -72, -81]
}
const red_piece_type = ['帅', '士', '相', '车', '马', '炮', '兵']
const black_piece_type = ['将', '仕', '象', '车', '马', '炮', '卒']
//只允许移动红方
const action_map_index = [
  0, 5, 4, 4, 4, 4, 34, 34, 8, 8, 34, 34, 3, 3, 3, 3, 3
]
const action_map_index_fc = () => {
  let L = [];
  let sum = 0;

  for (let i = 0; i < action_map_index.length; i++) {
    sum += action_map_index[i];
    L.push(sum);
  }

  return L;
};
const action_map3 = action_map_index_fc();

export function getXbyId(id) {
  return [action_map3[id], action_map3[id + 1] - 1];
}
export function getPieceTypeById(id) {
  let { color, piece, num } = id2piece(id);
  let index = red_piece_type.indexOf(piece);
  if (index == -1) {
    index = black_piece_type.indexOf(piece);
  }
  if (color == "black") {
    index += 7;
  }
  return index;
}
//人工编码状态，输出更强的语义信息
const H = 10;
const W = 9;
const C = 14;
const SIZE = H * W * C;

export function encode_state(state) {
  const buffer = new Float32Array(SIZE);

  for (let i = 0; i < state.length; i++) {
    const pos = state[i];
    if (pos === 90) continue;

    const [r, c] = num2rc(pos);
    const type = getPieceTypeById(i);

    const idx = (r * W + c) * C + type;

    buffer[idx] = 1;
  }

  return buffer;
}
