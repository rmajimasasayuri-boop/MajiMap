'use strict';

// Coordinates are in the original staff drawing. These are labels, pending doorway verification.
const staffPlanDestinations = [
  {floor:2, name:'2R 救命救急外来', pixel:[1030,904]},
  {floor:2, ref:'outpatient:2::2S', name:'2S 中央放射線部・放射線診断科', pixel:[1741,903], approach:[1742,942], mapLabel:'2S 受付'},
  {floor:2, code:'2S-9', name:'第1撮影室', pixel:[1395,1103]},
  {floor:4, name:'中央手術部（EV20側）', pixel:[1497,1134]},
  {floor:5, name:'CCMC', pixel:[1307,1090]},
  {floor:5, name:'ICU', pixel:[1700,1130]},
].map(({floor,ref,code,name,pixel,approach,mapLabel})=>({
  floor, ref, code, name, mapLabel, referenceOnly:true,
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
  ['18','X-TV',[1848,397]],
  ['19','第6撮影室',[1894,408]],
  ['20','第7撮影室',[1932,416]],
  ['21','第8撮影室',[1972,419]],
];
for (const [number,label,pixel,approach] of radiologyRooms) {
  const code='2S-'+number;
  staffPlanDestinations.push({
    id:code, floor:2, name:label, code, category:'radiology',
    point:staffPlanPoint(2,pixel), approach:approach && staffPlanPoint(2,approach), referenceOnly:true,
  });
}

for (const [id,name,pixel,approach] of [
  ['mri-toilet','第4MRI室横のトイレ',[1840,1014],[1839,959]],
  ['mri-ante-1','MRI前室1',[1702,1097],[1745,1093]],
  ['mri-ante-2','MRI前室2',[1788,1172],[1741,1113]],
  ['technologist','技師室',[1911,717],[1860,725]],
  ['reading','読影室',[1898,787],[1850,792]],
  ['operation-1','操作室1',[1458,1053],[1461,1050]],
  ['operation-2','操作室2',[1649,1115],[1649,1114]],
  ['operation-3','操作室3',[1842,1177],[1843,1169]],
  ['operation-4','操作室4',[1851,715],[1860,715]],
  ['operation-5','操作室5',[1882,368],[1881,365]],
]) {
  staffPlanDestinations.push({
    id:'2S-'+id, floor:2, name, mapLabel:id==='mri-toilet'?'トイレ':undefined,
    category:'radiology', aliases:'2S', referenceOnly:true,
    point:staffPlanPoint(2,pixel), approach:staffPlanPoint(2,approach),
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

// The access type comes from the corridor on the approach side of each door.
const radiologyEntrances = [
  ['099c87ed-5a05-44bf-a98f-e3c928789610', [
    ['2S-2','f93320ec-7353-445f-ac8f-1cd2438fcc3a'],
    ['2S-3','f29f539e-6e55-472c-9aad-de0a75f4ad82'],
    ['2S-mri-ante-1','0e8aecec-a6b3-4ed9-ae4f-e4f96224767d'],
  ]],
  ['d2533851-295b-4881-91e2-1f0518f78111', [
    ['2S-4','769ec754-b6d1-4aa9-b250-10ef8f15d03e'],
  ]],
  ['346a21f1-3fac-4cfd-9005-08a2d59946a3', [
    ['2S-5','10556df7-ee63-4e0f-99e6-2786884d7291'],
    ['2S-mri-ante-2','9bad690e-5f1a-4287-bac3-ab153aae4d11'],
  ]],
  ['f5dabe7c-06c4-4aa0-af32-d824f4d8b2dd', [
    ['2S-14','f99f46fb-fadd-4318-818c-a43755e3e725'],
    ['2S-15','f688e006-f66d-4ddc-b092-a39adec137b2'],
    ['2S-16','6a7659d9-d8ba-42de-a1b1-a539330de9be'],
    ['2S-17','28334a13-34a0-4618-a238-d77483ad0500'],
  ]],
  ['c8a3fecc-db9e-4dd6-967d-a96c2bb691ef', [
    ['2S-19','281894a8-043f-4f2f-98fe-cb8c95307271'],
    ['2S-20','90614702-b427-430b-98a3-78424f347a6e'],
    ['2S-21','70527308-2faa-498b-9fd4-9e2b0853be20'],
  ]],
  ['04dc7914-3381-4744-a505-dfa2bd4a29ab', [
    ['2S-reading','8201ff75-8f54-488d-baf3-8419b5b1df3e'],
    ['2S-technologist','5a522d3d-edce-41ec-a560-ebf9f8b17f1e'],
  ]],
  ['43a99009-8196-44ee-9a23-bb3c824ac822', [
    ['2S-operation-2','004c0dae-2ada-42a4-8559-b4e9ddbd2aca'],
  ]],
  ['7efd8b76-8c0f-4e7d-8fb6-c9641d90352d', [
    ['2S-operation-3','eadb5eb3-3204-43c8-ae36-3bf911702899'],
  ]],
  ['3affd2cc-8b69-4954-9c69-832b589619ab', [
    ['2S-mri-toilet','ce551989-3a54-4544-b51b-d9321a2f4722'],
    ['2S-operation-4','5b21db3d-e3f0-4859-addc-263fdf8cb6a4','職員用入口前'],
    ['2S-10','d9bd3497-ce93-420c-82b9-9200d78deb61'],
  ]],
  ['400ec46a-a640-40fc-9112-5defc21ede0c', [
    ['2S-operation-5','18613b63-9795-48ad-a9af-e22cf911ba5f','職員用入口前'],
    ['2S-18','20538e67-cd41-495a-86ae-af542cc6d007'],
    ['2S-11','4821bbe5-3e22-401d-99fd-b2a27c143fd2'],
    ['2S-12','3304cad6-8525-40ec-9bd9-4e4ba9c941e0'],
    ['2S-13','b300a059-cb9b-4e6c-b36b-c1906dc6aaa7'],
  ]],
];

const radiologyMapLabels = [
  ['更衣室',1782.3,645.4],
  ['器材室',1825.2,652.9],
  ['スタッフ室',1921,670.3],
  ['EPS4',1735.3,873.6],
];
