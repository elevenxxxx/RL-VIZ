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
  [85, 86, 84, 76, 77, 75, 67, 68, 66, 127],//红帅
  [127, 86, 84, 76, 68, 66],//红士1-右
  [127, 84, 86, 76, 66, 68],//红士2-左
  [127, 87, 67, 71, 51, 83, 47, 63],//红相1-右
  [127, 83, 67, 63, 47, 87, 51, 71],//红相2-左
  [127, ...RedAllCandidates],//红车1 0-89
  [127, ...RedAllCandidates],//红车2 0-89
  [127, ...RedAllCandidates],//红马1 0-89
  [127, ...RedAllCandidates],//红马2 0-89
  [127, ...RedAllCandidates],//红炮1 0-89
  [127, ...RedAllCandidates],//红炮2 0-89
  [127, 62, 53, ...RedSoliderCandidates],//红兵5 -最右
  [127, 60, 51, ...RedSoliderCandidates],//红兵4
  [127, 58, 49, ...RedSoliderCandidates],//红兵3
  [127, 56, 47, ...RedSoliderCandidates],//红兵2
  [127, 54, 45, ...RedSoliderCandidates],//红兵1 -最左
  [4, 3, 5, 13, 12, 14, 22, 21, 23, 127],//黑将
  [127, 3, 5, 13, 21, 23],//黑仕1-左
  [127, 5, 3, 13, 23, 21],//黑仕2-右
  [127, 2, 22, 18, 38, 6, 42, 26],//黑象1-左
  [127, 6, 22, 26, 42, 2, 38, 18],//黑象2-右
  [127, ...BlackAllCandidates],//黑车1 0-89
  [127, ...BlackAllCandidates],//黑车2 0-89
  [127, ...BlackAllCandidates],//黑马1 0-89
  [127, ...BlackAllCandidates],//黑马2 0-89
  [127, ...BlackAllCandidates],//黑炮1 0-89
  [127, ...BlackAllCandidates],//黑炮2 0-89
  [127, 27, 36, ...BlackSoliderCandidates],//黑兵5 -最左
  [127, 29, 38, ...BlackSoliderCandidates],//黑兵4
  [127, 31, 40, ...BlackSoliderCandidates],//黑兵3
  [127, 33, 42, ...BlackSoliderCandidates],//黑兵2
  [127, 35, 44, ...BlackSoliderCandidates],//黑兵1 -最右
];
export const initMap=[
  85,86,84,87,83,
  81,89,82,88,64,70,
  62,60,58,56,54,
  4,3,5,2,6,
  0,8,1,7,19,25,
  27,29,31,33,35
];
const validarr=new Array(90).fill(0);
export const validMap = (arr) => {
    validarr.fill(0);

    for (let item of arr) {
        if (item === 127) continue;
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
  const array= [];
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
