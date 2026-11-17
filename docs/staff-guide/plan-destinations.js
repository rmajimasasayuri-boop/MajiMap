'use strict';

// Coordinates are in the original staff drawing. These are labels, pending doorway verification.
const staffPlanDestinations = [
  {floor:2, name:'2R 救命救急外来', pixel:[1030,904]},
  {floor:2, ref:'outpatient:2::2S', name:'2S 中央放射線部・放射線診断科', pixel:[1730,891], approach:[1742,942]},
  {floor:2, name:'第1撮影室（2S-9）', pixel:[1395,1103]},
  {floor:4, name:'中央手術部（EV20側）', pixel:[1497,1134]},
  {floor:5, name:'CCMC', pixel:[1307,1090]},
  {floor:5, name:'ICU', pixel:[1700,1130]},
].map(({floor,ref,name,pixel,approach})=>({
  floor, ref, name, referenceOnly:true,
  approach:approach && staffPlanPoint(floor,approach),
  point:staffPlanPoint(floor,pixel),
}));

function staffPlanPoint(floor, [x,y]) {
  const {width,height}=staffData.plans[floor];
  return [x*1000/width,y*1000/height];
}

// The approach in each row sits on the annotated corridor on the door's approach side.
const radiologyRooms = [
  ['1','診察室',[1788,923],[1759.3,944.9]],
  ['2','第1MRI室',[1639,1168],[1732.2,1164.6]],
  ['3','第2MRI室',[1661,1037],[1744.8,1093.5]],
  ['4','第3MRI室',[1831,1229],[1727.6,1190.7]],
  ['5','第4MRI室',[1849,1102],[1741.4,1112.8]],
  ['10','第1CT室',[1883,903],[1833.4,865.4]],
  ['11','第2CT室',[1774,833],[1841.5,818.9]],
  ['12','処置室',[1778,770],[1849.1,776.1]],
  ['13','第3CT室',[1794,704],[1855.8,737.9]],
  ['14','第2撮影室',[1798,568],[1775.3,611.4]],
  ['15','第3撮影室',[1842,578],[1858.3,626]],
  ['16','第4撮影室',[1894,587],[1872.1,628.5]],
  ['17','第5撮影室',[1941,595],[1920.1,636.9]],
];
for (const [number,label,pixel,approach] of radiologyRooms) {
  const code='2S-'+number;
  staffPlanDestinations.push({
    id:code, floor:2, name:number==='13'?label:code+' '+label, code, category:'radiology',
    point:staffPlanPoint(2,pixel), approach:staffPlanPoint(2,approach), referenceOnly:true,
  });
}

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

// Door IDs from the 2F review and the patient corridors those doors face.
const radiologyPatientEntrances = [
  ['3affd2cc-8b69-4954-9c69-832b589619ab', [
    ['2S-10','d9bd3497-ce93-420c-82b9-9200d78deb61'],
  ]],
  ['400ec46a-a640-40fc-9112-5defc21ede0c', [
    ['2S-11','4821bbe5-3e22-401d-99fd-b2a27c143fd2'],
    ['2S-12','3304cad6-8525-40ec-9bd9-4e4ba9c941e0'],
    ['2S-13','b300a059-cb9b-4e6c-b36b-c1906dc6aaa7'],
  ]],
];
