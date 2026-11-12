// Keep the exact recorded geometry; group small drawing offsets without announcing false turns.
function turnWord(a,b,c){const u=[b[0]-a[0],b[1]-a[1]],v=[c[0]-b[0],c[1]-b[1]],cross=u[0]*v[1]-u[1]*v[0],dot=u[0]*v[0]+u[1]*v[1];return Math.abs(cross)<1e-5?(dot<0?'折り返して':'まっすぐ'):cross>0?'右折して':'左折して';}
// Group the bends along the recorded 2F convenience-store approaches.
function sevenApproachCorners(floor,points){
 if(floor!==2)return [];
 const close=(a,b)=>Math.hypot(a[0]-b[0],a[1]-b[1])<.2;
 const endpoints=[points[0],points.at(-1)];
 const indices=(...corners)=>corners.map(c=>points.findIndex(p=>close(p,c))).filter(i=>i>0&&i<points.length-1);
 if(endpoints.some(p=>close(p,[745.3,691.1])))return indices([780,691.1],[780,727]);
 if(endpoints.some(p=>close(p,[676.6,724]))&&endpoints.some(p=>close(p,[570,679])))return indices([611,679],[611,727]);
 return [];
}
// These map points were confirmed as parts of a continuous walking instruction.
const walkingPassThroughPoints={
 3:[[302.4,308]], // 3B: short approach from the corridor to the clinic.
 5:[[600,962],[1530,470]], // Outpatient EV approach and the bend beside the HCU entrance.
};
function walkingBranchPoints(step){
 const graph=mapUI.graph,nodes=step.nodes||[];
 if(!graph||nodes.length<3)return [];
 const start=step.routeStart||nodes[0],end=step.routeEnd||nodes.at(-1);
 const via3F=routeNeeds3F(graph,start,end);
 const allowedWalk=(from,edge)=>edge.kind==='walk'&&referenceFloor(edge.to)===step.floor&&routeEdgeAllowed(graph,from,edge,via3F,start,end);
 const firstDirection=(from,to)=>{
  const points=squareRoute(routeGeometry(graph,from,to));
  if(points.length<2)return '';
  const [a,b]=points;
  return Math.sign(b[0]-a[0])+','+Math.sign(b[1]-a[1]);
 };
 const passThrough=walkingPassThroughPoints[step.floor]||[];
 const junctions=[];
 for(let first=0;first<nodes.length;){
  const point=displayPoint(nodes[first]);
  let after=first+1;
  // Some route edges join aliases drawn at the same point.
  while(point&&after<nodes.length){
   const nextPoint=displayPoint(nodes[after]);
   if(!nextPoint||Math.hypot(point[0]-nextPoint[0],point[1]-nextPoint[1])>=.01)break;
   after++;
  }
  if(point&&first>0&&after<nodes.length&&!passThrough.some(([x,y])=>Math.hypot(x-point[0],y-point[1])<.1)){
   const group=nodes.slice(first,after),previous=nodes[first-1],next=nodes[after];
   const used=new Set([firstDirection(group[0],previous),firstDirection(group.at(-1),next)]);
   if(!used.has('')&&used.size===2&&group.some(id=>(graph.adj[id]||[]).some(edge=>{
    if(group.includes(edge.to)||edge.to===previous||edge.to===next||!allowedWalk(id,edge))return false;
    const direction=firstDirection(id,edge.to);
    if(!direction||used.has(direction))return false;
    // A destination spur alone does not provide another through corridor.
    return (graph.adj[edge.to]||[]).some(onward=>!group.includes(onward.to)&&allowedWalk(edge.to,onward));
   })))junctions.push(point);
  }
  first=after;
 }
 return junctions;
}
function walkingInstructionParts(step){
 const source=instructionSegments(step),runs=[];
 for(const segment of source){
  if(!segment.points.length)continue;
  let run=runs.at(-1);
  if(!run||Math.hypot(run.at(-1)[0]-segment.points[0][0],run.at(-1)[1]-segment.points[0][1])>.01){
   run=[];runs.push(run);
  }
  for(const point of segment.points){
   if(!run.length||Math.hypot(run.at(-1)[0]-point[0],run.at(-1)[1]-point[1])>.001)run.push(point);
  }
 }
 const branches=walkingBranchPoints(step);
 // Thresholds are in display-map coordinates. Keep the full path for arrows and progress.
 const drawingOffset=6,shortApproach=32,evApproach=40;
 const result=[];
 for(const raw of runs){
  const ps=[];
  for(const p of raw){
   while(ps.length>1&&turnWord(ps.at(-2),ps.at(-1),p)==='まっすぐ')ps.pop();
   ps.push(p);
  }
  if(ps.length<2)continue;
  const length=i=>Math.hypot(ps[i+1][0]-ps[i][0],ps[i+1][1]-ps[i][1]);
  const minor=new Set(sevenApproachCorners(step.floor,ps));
  if(length(0)<(step.startsAtEV?evApproach:shortApproach))minor.add(1);
  if(length(ps.length-2)<(step.endsAtEV?evApproach:shortApproach))minor.add(ps.length-2);
  for(let i=1;i<ps.length-2;i++){
   const [a,b,c,d]=ps.slice(i-1,i+3);
   const u=[b[0]-a[0],b[1]-a[1]],v=[d[0]-c[0],d[1]-c[1]];
   if(length(i)<evApproach&&Math.abs(u[0]*v[1]-u[1]*v[0])<.001&&u[0]*v[0]+u[1]*v[1]>0){
    minor.add(i);minor.add(i+1);
   }
  }
  let begin=0,action='まっすぐ',hasBend=false;
  for(let i=1;i<ps.length;i++){
   const last=i===ps.length-1;
   const turn=last?'まっすぐ':turnWord(ps[i-1],ps[i],ps[i+1]);
   const [x,y]=ps[i];
   const corner=turn!=='まっすぐ'&&!minor.has(i)&&length(i-1)>=drawingOffset&&length(i)>=drawingOffset&&
    branches.some(([bx,by])=>Math.hypot(bx-x,by-y)<.01);
   if(!last&&!corner){
    hasBend||=turn!=='まっすぐ';
    continue;
   }
   const maneuver=action==='まっすぐ'&&hasBend?'通路に沿って':action;
   const target=last?(step.endLabel||'この区間の終点'):'地図に示した分岐';
   const title=!last&&maneuver==='まっすぐ'?target+'まで直進':maneuver+target+'まで進む';
   result.push({...step,title,description:step.description?.includes('要確認')?'要確認箇所を含みます。青線に沿って進んでください。':'地図の橙色の丸が、この手順で進む位置です。通路に沿ってそこまで進んでください。',segments:[{floor:step.floor,points:ps.slice(begin,i+1)}],maneuver});
   begin=i;action=turn;hasBend=false;
  }
 }
 const portals=[...(typeof confirmedNavigation!=='undefined'?confirmedNavigation.portals||[]:[]),...(typeof additionalNavigation!=='undefined'?additionalNavigation.portals||[]:[]),...(typeof mobilityData!=='undefined'?mobilityData.portals||[]:[])];
 for(const [index,part]of result.entries()){const points=part.segments[0].points,length=points.slice(1).reduce((s,p,i)=>s+Math.hypot(p[0]-points[i][0],p[1]-points[i][1]),0);for(const portal of portals){if(portal.floor!==step.floor||portal.arrivalOnly&&(index!==result.length-1||step.nodes.at(-1)!==portal.id))continue;const near=nearestOnPolyline(portal.point,points);if(!near||near.distance>6||/EV|エレベーター/i.test(portal.label)&&((step.startsAtEV&&near.along<40)||(step.endsAtEV&&length-near.along<40)))continue;const through=!portal.arrivalOnly&&length-near.along>=10;part.description+=' '+portal.label+(through?'を通って進んでください。':'まで進んでください。');if(through&&!part.title.includes('出入口'))part.title+='（出入口を通る）';}}
 return result.length?result:[step];
}
