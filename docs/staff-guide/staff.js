'use strict';

const staffPlans = staffData.plans;
let staffRecords = staffData.records;
let staffRouteStatus = '';
let staffRouteEndpoints = null;
const staffPoints = new Map();
const staffAdded = new Set();
const staffPixels = (f, p) => [p[0] * staffPlans[f].width / 1000, p[1] * staffPlans[f].height / 1000];
const staffNormalized = (f, p) => [p[0] * 1000 / staffPlans[f].width, p[1] * 1000 / staffPlans[f].height];

function loadStaffLocations() {
  for (let i = destinationList.length - 1; i >= 0; i--) {
    if (staffAdded.has(destinationList[i].ref)) destinationList.splice(i, 1);
  }
  for (const option of [...ui('fromSelect').options]) if (staffAdded.has(option.value)) option.remove();
  staffAdded.clear();
  staffPoints.clear();
  const locations = [...staffData.places, ...staffPlanDestinations, ...staffRecords
    .filter(r => ['stairs','elevator'].includes(r.kind))
    .map(r => ({...r, point:r.points[0]}))];
  for (const [index, p] of locations.entries()) {
    const match = p.ref ? destinationList.find(d => d.ref===p.ref) : !p.referenceOnly && !p.category && destinationList.find(d => d.floor === p.floor &&
      (d.code && p.name.startsWith(d.code + ' ') ||
       d.ref === 'hospital:3::generalReception' && p.name.startsWith('総合受付') ||
       d.ref === 'hospital:7::rehab' && p.name.includes('リハビリ')));
    const liftInfo = p.kind==='elevator' && staffData.liftInfo?.[p.name];
    const ref = match?.ref || 'hospital:' + p.floor + '::staff-' + (p.id || index);
    if (!match) {
      destinationList.push({ref, floor:p.floor, name:p.referenceOnly?p.name+'（入口未確認・付近まで）':liftInfo?p.name+' '+liftInfo.purpose:p.name, code:p.code || (p.kind==='elevator'?p.name:''),
        type:'staff-location', detail:{building:'病院棟',aliases:liftInfo?.stops}});
      staffAdded.add(ref);
    }
    staffPoints.set(ref, p);
  }
  for (const d of destinationList.filter(d=>staffAdded.has(d.ref))) ui('fromSelect').add(new Option(d.floor+'階 '+d.name,d.ref));
}

for (const [floor, plan] of Object.entries(staffPlans)) {
  Object.assign(unifiedMapModel.floors[floor], {
    image:'staff-guide/maps/' + floor + '-verified-plan.webp',
    size:[plan.width,plan.height],
  });
}

const patientShowFloor = showFloor;
showFloor = f => {
  patientShowFloor(f, false);
  const background = ui('mapBackground');
  background.alt = mapUI.floor + '階の' + '案内マップ';
  ui('baseBadge').textContent = mapUI.floor + 'F';
};
visitorCropActive = () => false;
focusWard = () => {
  if (mapUI.floor < 7) return;
  const f = mapUI.floor, plan = staffPlans[f], viewport = ui('mapViewport');
  const [x,y,w,h] = f === 7 ? [100,240,780,510] : plan.view;
  const scale = Math.min((viewport.clientWidth-36)/w,(viewport.clientHeight-55)/h);
  mapUI.zoom = Math.max(1,scale/mapUI.base);
  mapUI.pan = [(plan.width/2-x-w/2)*scale,(plan.height/2-y-h/2)*scale];
  applyMapTransform();
};
ui('wardFocus').onclick = () => focusWard();

function staffLocation(d) {
  const p = staffPoints.get(d?.ref);
  return p ? staffPixels(p.floor,p.point) : d?.type==='manual' ? d.point : null;
}
displayPoint = ref => mapUI.graph?.nodes[ref]?.displayPoint || staffLocation(destinationList.find(d => d.ref===ref));
navigationTargetReference = d => d.ref;
isGuideOnly = d => !staffLocation(d);
routeStartPoint = () => (mapUI.path.length && staffRouteEndpoints?.start) || mapUI.manual?.point || displayPoint(mapUI.from);
routeGoalPoint = () => (mapUI.path.length && staffRouteEndpoints?.goal) || mapUI.manualGoal?.point || staffLocation(mapUI.goal);
const patientResultCategoryMatches = resultCategoryMatches;
resultCategoryMatches = d => d.type==='staff-location'
  ? mapUI.filter==='all' || (mapUI.filter==='equipment' && Boolean(staffPoints.get(d.ref)?.kind))
  : patientResultCategoryMatches(d);
routeProximityFor = () => new Map();
squareRoute = points => points;
canDrawFloorRoute = () => true;

function staffRegions(d) {
  const p = staffLocation(d);
  return p ? [[p[0]-16,p[1]-11,32,22]] : [];
}
placeHitBounds = staffRegions;
placeHitPolygons = d => staffRegions(d).map(([x,y,w,h]) => [[x,y],[x+w,y],[x+w,y+h],[x,y+h]]);
placeLabelFrames = staffRegions;
wardCourtyardPolygons = () => [];
placeAtMapPoint = (places, floor, point) => {
  let nearest, min = Math.max(15,12/(mapUI.base*mapUI.zoom||1));
  for (const place of places) {
    if (place.floor!==floor) continue;
    const p = staffLocation(place);
    if (!p) continue;
    const distance = Math.hypot(point[0]-p[0],point[1]-p[1]);
    if (distance < min) { min = distance; nearest = place; }
  }
  return nearest;
};

function staffPolyline(points, color, width, dash='') {
  return svgNode('polyline',{points:points.map(p=>p.join(',')).join(' '),fill:'none',stroke:color,
    'stroke-width':width,'stroke-dasharray':dash,'stroke-linecap':'round','stroke-linejoin':'round',
    'vector-effect':'non-scaling-stroke','pointer-events':'none'});
}
// Room labels sit inside the room or ward. The map marker stays at its entrance.
function staffLabelPosition(d, place, x, y) {
    if (place.category==='conference') return {x,y:y+24,'text-anchor':'middle'};
    if (d.floor>=7 && /病棟/.test(d.name)) {
      const ward = (d.code+' '+d.name).match(/(東|西)(A|B)/);
      if (ward) return {
        x:x+(ward[1]==='東'?96:-96),
        y:y+(ward[2]==='A'?-185:170),
        'text-anchor':'middle',
      };
    }
    const scale = mapUI.base * mapUI.zoom || 1;
    return {x:x+9/scale,y:y-6/scale};
}
renderOverlay = () => {
  const svg = ui('unifiedOverlay'), f = mapUI.floor;
  svg.replaceChildren();
  for (const line of floorRouteLines(mapUI.graph,mapUI.path,f)) {
    svg.append(staffPolyline(line,'white',8),staffPolyline(line,'#0878d1',4.5,staffRouteStatus?'7 3':''));
  }
  for (const d of destinationList) {
    if (d.floor!==f) continue;
    const position = staffLocation(d);
    if (!position) continue;
    const [x,y] = position, kind = staffPoints.get(d.ref)?.kind;
    const place = staffPoints.get(d.ref);
    const conference = place?.category==='conference';
    const ward = d.name.includes('病棟');
    const primary = conference || ward || d.ref==='outpatient:2::2S' || /リハビリ|EVホール|総合受付/.test(d.name);
    const color = conference?'#7b3e9d':ward?'#177255':kind==='stairs'?'#564684':'#176489';
    const target = svgNode('g',{class:'staff-marker',role:'button',tabindex:0,'aria-label':d.name,'data-place-ref':d.ref});
    const liftInfo = kind==='elevator' && staffData.liftInfo?.[place.name];
    target.append(svgNode('title',{},liftInfo?d.name+' · '+liftInfo.stops:d.name));
    target.append(svgNode('circle',{cx:x,cy:y,r:11,fill:'transparent'}));
    const zoomScale = mapUI.base * mapUI.zoom || 1;
    if (kind==='elevator' && !primary) {
      const iconSize = 16/zoomScale;
      target.append(svgNode('use',{href:'assets/equipment-pictograms.svg#elevator',
        x:x-iconSize/2,y:y-iconSize/2,width:iconSize,height:iconSize,'pointer-events':'none'}));
    } else if (kind==='stairs') target.append(svgNode('polygon',{points:`${x},${y-8} ${x-8},${y+6} ${x+8},${y+6}`,fill:color,stroke:'white'}));
    else target.append(svgNode('circle',{cx:x,cy:y,r:primary?9:5,fill:color,stroke:'white','stroke-width':2}));
    const showLabel = primary || (place.category==='radiology'&&mapUI.zoom>=1.65) || (kind==='stairs'&&mapUI.zoom>=1.4) || mapUI.zoom>=2;
    if (showLabel) {
      const label = conference ? '会議室 ' + d.code : d.code || d.name;
      const position = staffLabelPosition(d,place,x,y);
      target.append(svgNode('text',{...position,fill:color,'font-size':(primary?14:11)/zoomScale,
        'font-weight':'bold',stroke:'white','stroke-width':2.8/zoomScale,'paint-order':'stroke','pointer-events':'none'},label));
    }
    target.onkeydown = e => {
      if (e.key==='Enter'||e.key===' ') { e.preventDefault(); e.stopPropagation(); previewPlacePhoto(d,null,true); }
    };
    target.onclick = e => { if (e.detail===0) previewPlacePhoto(d,null,true); };
    svg.append(target);
  }
  const from = routeStartPoint(), to = routeGoalPoint();
  if (hasVisibleStart()&&!navigationState.active&&referenceFloor(mapUI.from)===f&&from) addPin(from,'出発地','#2580c5');
  if (mapUI.goal?.floor===f&&to) {
    const nearby = mapUI.path.length && staffPoints.get(mapUI.goal.ref)?.referenceOnly;
    addPin(to,nearby?'付近の通路':'目的地','#d83a40');
  }
  drawNavigationPosition();
  window.dispatchEvent(new Event('maji-map-update'));
};

const walkChoice = document.querySelector('[data-travel-choice="auto"]');
for (const label of walkChoice.childNodes) {
  if (label.nodeType===Node.TEXT_NODE&&label.textContent.includes('通常')) label.textContent=label.textContent.replace('通常','徒歩');
}
const stretcherChoice = document.createElement('button');
stretcherChoice.type='button';
stretcherChoice.dataset.travelChoice='stretcher';
stretcherChoice.textContent='ストレッチャー';
stretcherChoice.onclick=()=>{
  ui('travelChoice').value='stretcher';
  ui('travelChoice').onchange();
  updateRoutePlanner();
};
document.querySelector('.route-travel-tabs').append(stretcherChoice);

const staffOptions = document.createElement('div');
staffOptions.className='staff-options';
staffOptions.innerHTML='<label id="staffLiftField" hidden>使用EV <select id="staffLift"><option value="">搬送ルールに従う</option></select></label><p id="staffStatus" role="status" hidden></p>';
staffOptions.addEventListener('pointerdown', event => event.stopPropagation());
ui('routeOptionsDock').append(staffOptions);
for (const lift of staffData.rules.stretcherElevators) ui('staffLift').add(new Option(lift,lift));
ui('staffLift').onchange = () => routePlan();
const patientUpdatePlanner = updateRoutePlanner;
updateRoutePlanner = () => {
  patientUpdatePlanner();
  ui('staffLiftField').hidden=ui('travelChoice').value!=='stretcher';
  staffOptions.hidden=ui('staffLiftField').hidden&&ui('staffStatus').hidden;
};
window.addEventListener('maji-map-update',updateRoutePlanner);

const staffMapControls = document.createElement('div');
staffMapControls.className = 'staff-map-controls';
staffMapControls.setAttribute('aria-label', '職員マップの表示');
const staffLegendButton = document.createElement('button');
staffLegendButton.type = 'button';
staffLegendButton.textContent = '地図の凡例';
staffLegendButton.onclick = () => ui('mapInfoDialog').showModal();
for (const [label, query] of [['病棟','病棟'], ['カンファレンス室','カンファレンス室'], ['2S周辺','2S']]) {
  const button = document.createElement('button');
  button.type = 'button';
  button.textContent = label;
  button.onclick = () => {
    ui('openPlaceSearch').click();
    const search = ui('findPlace');
    search.value = query;
    search.dispatchEvent(new Event('input', {bubbles:true}));
    search.dispatchEvent(new KeyboardEvent('keydown', {key:'Enter',code:'Enter',bubbles:true}));
  };
  staffMapControls.append(button);
}
staffMapControls.append(staffLegendButton);
ui('mapViewport').append(staffMapControls);

const staffLegend=document.createElement('p');
staffLegend.id='staffLegend';
staffLegend.textContent='緑は病棟入口、紫はカンファレンス室です。EVの用途・停止階はEV番号・用途図と照合しています。青い実線は案内経路、青い破線は試算経路です。入口未確認の施設は、登録された近くの通路まで案内します。';
ui('mapInfoDialog').prepend(staffLegend);
function staffNotice(message) {
  const node=ui('staffStatus');
  node.textContent=message;
  node.hidden=!message;
  if(message) ui('selectionHint').textContent=message;
  else if(!mapUI.selectingStartOnMap) ui('selectionHint').textContent='';
}
function staffEndpoint(ref, manual) {
  if (manual) return {id:ref, floor:manual.floor, name:manual.name,
    point:staffNormalized(manual.floor,manual.point)};
  const d=destinationList.find(d=>d.ref===ref), p=staffPoints.get(ref);
  return p&&{...p,point:p.approach||p.point,id:ref,name:p.referenceOnly?p.name:
    d.code&&!d.name.includes(d.code)?d.code+' '+d.name:d.name};
}
function requiredStaffLift(start, goal) {
  const names=[start.name,goal.name], ward={'西A':'EV9','西B':'EV8','東A':'EV11','東B':'EV7'};
  const intensive=names.some(n=>/CCMC|ICU/i.test(n));
  if(intensive&&names.some(n=>n.includes('手術室')))return 'EV20';
  if(intensive&&names.some(n=>/第[1１]撮影室/.test(n))||
    names.some(n=>n.includes('2R'))&&[start.floor,goal.floor].some(f=>f===5||f===6))return 'EV16';
  return ui('staffLift').value||ward[names.slice().reverse().map(n=>n.match(/西A|西B|東A|東B/)?.[0]).find(Boolean)];
}
routePlan = () => {
  staffRouteStatus='';
  staffRouteEndpoints=null;
  staffNotice('');
  navigationState.steps=[];
  mapUI.path=[];
  mapUI.graph=null;
  ui('routeJourney').replaceChildren();
  ui('routeSummary').hidden=true;
  ui('beginNavigation').disabled=true;
  mapUI.routeArrival=null;
  mapUI.routeDeparture=null;
  if (!mapUI.goal||!ensureIndoorStart(false)) return renderOverlay();
  const start=staffEndpoint(mapUI.from,mapUI.manual), goal=staffEndpoint(mapUI.goal.ref,mapUI.manualGoal);
  const mode=({auto:'walk',elevator:'wheelchair',stretcher:'stretcher'})[ui('travelChoice').value];
  const provisional = Boolean(start?.referenceOnly || goal?.referenceOnly);
  const result=start&&goal&&start.id!==goal.id
    ? staffRoute(staffRecords,staffPlans,start,goal,mode,true,requiredStaffLift(start,goal))
    : {error:start&&goal?'出発地と目的地は同じ場所です。':'この施設の位置は未登録です。地図上で出発地・目的地を指定してください。'};
  if (result.error) {staffNotice(result.error);return renderOverlay();}
  staffRouteEndpoints = {start:result.path[0].p, goal:result.path.at(-1).p};
  const entranceNotice = goal.referenceOnly ? '入口未確認 · 目的地付近の通路まで'
    : start.referenceOnly ? '入口未確認 · 出発地付近の通路から' : '';
  staffRouteStatus = [entranceNotice, result.uncertain && '試算 · 未確認区間を含みます']
    .filter(Boolean).join('。');
  const graph={nodes:{},adj:{}}, refs=result.path.map((n,i)=>`hospital:${n.floor}::staff-route-${i}`);
  let run=[], transferIndex=0;
  function addWalk(points, f, title) {
    if (points.length) navigationState.steps.push({floor:f,kind:'walk',title,
      description:staffRouteStatus||'地図の青線に沿って進んでください。',segments:[{floor:f,points:points.slice()}]});
  }
  for (const [i,node] of result.path.entries()) {
    const id=refs[i];
    graph.nodes[id]={map:'hospital:'+node.floor,displayPoint:node.p};
    graph.adj[id]=[];
    const prior=result.path[i-1];
    if (prior) {
      const from=refs[i-1], changesFloor=prior.floor!==node.floor;
      const transfer=changesFloor?result.transfers[transferIndex++]:null;
      const kind=transfer?(transfer.name.startsWith('EV')?'elevator':'stairs'):'walk';
      const edge={kind,u:from,v:id,...(!changesFloor?{displayPath:[prior.p,node.p]}:{})};
      graph.adj[from].push({...edge,to:id});
      graph.adj[id].push({...edge,to:from});
      if (changesFloor) {
        addWalk(run,prior.floor,transfer.name+'まで');
        run=[];
        const title=`${transfer.name}で ${transfer.from}階 → ${transfer.to}階`;
        if (kind==='elevator') navigationState.steps.push({floor:node.floor,kind:'transfer',from,to:id,title});
        else addWalk([prior.p],prior.floor,title);
      }
    }
    run.push(node.p);
  }
  addWalk(run,goal.floor,goal.name+(goal.referenceOnly?'付近の通路':goal.approach?'の入口手前':'')+'まで');
  if(result.uncertain || provisional) for(const step of navigationState.steps) step.title='試算｜'+step.title;
  mapUI.graph=graph;
  mapUI.path=refs;
  for(const step of navigationState.steps){
    const button=document.createElement('button');
    button.textContent=step.title;
    button.onclick=()=>showFloor(step.floor);
    ui('routeJourney').append(button);
  }
  ui('routeSummary').hidden=false;
  ui('routeSummaryText').textContent=staffRouteStatus||start.name+' → '+goal.name;
  ui('beginNavigation').disabled=false;
  staffNotice(staffRouteStatus);
  renderOverlay();
};

selectMapPoint = point => {
  if(arrowEditor.active||corridorEditor.active||!mapUI.selectingEndpointOnMap)return;
  const start=mapUI.selectingEndpointOnMap==='start';
  const f=mapUI.floor;
  const mode=({auto:'walk',elevator:'wheelchair',stretcher:'stretcher'})[ui('travelChoice').value];
  let nearest, distance=20/(mapUI.base*mapUI.zoom||1);
  for (const r of staffRecords){
    if(r.kind!=='corridor'||r.floor!==f||r.modes[mode]==='no')continue;
    const hit=nearestOnPolyline(point,r.points.map(p=>staffPixels(f,p)));
    if(hit&&hit.distance<distance){nearest=hit;distance=hit.distance;}
  }
  if(!nearest){staffNotice('通路上の位置か、地図の施設マークを選んでください。');return;}
  const named=window.prompt(start?'出発地の名前（任意）':'目的地の名前（任意）');
  if(named===null)return;
  mapUI.selectingEndpointOnMap=null;
  const ref='hospital:'+f+'::__manual'+(start?'Start':'Goal');
  const name=named.trim()||(start?'地図で指定した出発地':'地図で指定した目的地');
  const location={ref, floor:f, point:nearest.point, name, code:'', type:'manual'};
  if(start){
    selectStartPlace(location,true);
    mapUI.manual=location;
    mapUI.startName=name;
    updateCurrentLabel();
  }else{
    choosePlace(location,true);
    mapUI.manualGoal=location;
  }
  enterRoutePlanning();
};

for (const id of ['annToggle','toggleArrowEditor','editMedicineEVMain']) {
  const control = ui(id);
  if (control) {
    control.onclick = null;
    control.style.display = 'none';
    control.disabled = true;
  }
}
loadStaffLocations();
showFloor(mapUI.floor);
renderResults();

const staffQuery = new URLSearchParams(location.search);
const staffLinkedStart = destinationList.find(d=>d.ref===staffQuery.get('from'));
const staffLinkedGoal = destinationList.find(d=>d.ref===staffQuery.get('goal'));
if (staffQuery.get('mode')==='stretcher') {
  ui('travelChoice').value='stretcher';
  ui('travelChoice').onchange();
}
if (staffQuery.has('from')) mapUI.startConfirmed=false;
if (staffLinkedStart&&staffPoints.has(staffLinkedStart.ref)) selectStartPlace(staffLinkedStart);
else if (staffQuery.has('from')) updateCurrentLabel();
if (staffLinkedGoal&&staffPoints.has(staffLinkedGoal.ref)) {
  choosePlace(staffLinkedGoal);
  enterRoutePlanning();
}

if (staffQuery.get('find')==='カンファレンス室') {
  ui('openPlaceSearch').click();
  const search = ui('findPlace');
  search.value = staffQuery.get('find');
  search.dispatchEvent(new Event('input', {bubbles:true}));
  search.dispatchEvent(new KeyboardEvent('keydown', {key:'Enter',code:'Enter',bubbles:true}));
}
