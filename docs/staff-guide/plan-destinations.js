'use strict';

// Coordinates are in the original staff drawing. These are labels, pending doorway verification.
const staffPlanDestinations = [
  {floor:2, name:'2R 救命救急外来', pixel:[1030,904]},
  {floor:2, name:'第1撮影室（2S-9）', pixel:[1395,1103]},
  {floor:4, name:'中央手術部（EV20側）', pixel:[1497,1134]},
  {floor:5, name:'CCMC', pixel:[1307,1090]},
  {floor:5, name:'ICU', pixel:[1700,1130]},
].map(({floor,name,pixel})=>({
  floor, name, referenceOnly:true,
  point:[pixel[0] * 1000 / staffData.plans[floor].width, pixel[1] * 1000 / staffData.plans[floor].height],
}));

// Room labels use the aligned hospital drawings. Approach points are on the recorded corridor outside the door.
const conferenceRooms = [
  [1, '1Y-1', [1052, 780]],
  [1, '1Y-2', [1853, 1097]],
  [1, '1Y-3', [1726, 357]],
  [1, '1X-1', [587, 820]],
  [2, '2Y-1', [1480, 532]],
  [2, '2Y-2', [1760, 350]],
  [2, '2Y-3', [1889, 843]],
  [2, '2X-1', [425, 802]],
  [2, '2X-2', [217, 979]],
  [3, '3X-1', [598, 769], [548, 759]],
  [4, '4Y-3', [1526, 370]],
  [4, '4Y-1', [1460, 835]],
  [4, '4Y-2', [1615, 445]],
  [4, '4X-1', [598, 752], [547, 745]],
  [5, '5Y-2', [1905, 702]],
  [5, '5Y-4', [1113, 1026]],
  [5, '5Y-1', [1567, 1040]],
  [5, '5Y-3', [1216, 527]],
  [5, '5Y-5', [1589, 494]],
  [5, '5X-1', [598, 771]],
  [6, '6Y-3', [1007, 528]],
  [6, '6Y-4', [1265, 869]],
  [6, '6Y-1', [1709, 384]],
  [6, '6Y-2', [1714, 346]],
  [7, '7Y-2', [165, 506]],
  [7, '7Y-1', [792, 521], [822, 570]],
  [8, '8Y-1', [793, 520], [823, 570]],
  [8, '8Y-2', [199, 466]],
  [9, '9Y-1', [867, 520], [897, 570]],
  [9, '9Y-2', [277, 467]],
  [10, '10Y-2', [269, 463]],
  [10, '10Y-1', [864, 518], [894, 570.61578]],
];

for (const [floor, code, pixel, approach] of conferenceRooms) {
  const plan = staffData.plans[floor];
  const normalize = ([x,y]) => [x * 1000 / plan.width, y * 1000 / plan.height];
  staffPlanDestinations.push({
    id:'conference-' + code, floor, code, name:'カンファレンス室 ' + code,
    category:'conference', point:normalize(pixel),
    approach:approach && normalize(approach), referenceOnly:!approach,
  });
}
