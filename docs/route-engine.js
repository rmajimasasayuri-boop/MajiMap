const $=id=>document.getElementById(id);
Object.assign(floorMaps,outpatientMaps);
const buildings={hospital:{title:'病棟',defaultFloor:7},outpatient:{title:'外来・診療棟',defaultFloor:3}};
let travelMode='elevator',destinationCategory='all';
function serviceGoal(){const p=splitRef(selectedGoal);return p.map==='services'?serviceDestinations[p.id]:null}
function destinationType(key,id){if(key==='services')return 'services';if(key==='hospital:3'&&id==='generalReception')return 'reception';if(floorMaps[key]?.destinationDetails?.[id]?.arrivalKind==='areaEntrance')return 'clinic';if(['eastA','eastB','westA','westB'].includes(id))return 'ward';return 'other'}
function matchesCategory(key,id){return destinationCategory==='all'||destinationType(key,id)===destinationCategory}
function goalDetails(){const p=splitRef(selectedGoal);return floorMaps[p.map]?.destinationDetails?.[p.id]}
function isClinicGoal(){return goalDetails()?.arrivalKind==='areaEntrance'}
function patientDestination(key,id){const m=floorMaps[key];return m?.places.some(p=>p[0]===id)&&((m.buildingKey==='outpatient'&&m.destinationDetails?.[id]?.arrivalKind==='areaEntrance')||(key==='hospital:3'&&id==='generalReception')||(m.buildingKey==='hospital'&&['eastA','eastB','westA','westB'].includes(id)));}
function destinationAllowed(key,id){if(key==='services')return !!serviceDestinations[id];return audience==='staff'?floorMaps[key]?.places.some(p=>p[0]===id):patientDestination(key,id);}
function destinationCount(){return Object.keys(serviceDestinations).length+Object.entries(floorMaps).reduce((n,[key,m])=>n+m.places.filter(p=>destinationAllowed(key,p[0])).length,0);}
const ref=(map,id)=>map+'::'+id;
function splitRef(value){const i=value.indexOf('::');return{map:value.slice(0,i),id:value.slice(i+2)}}
function mapName(key){const m=floorMaps[key];return m.title+' '+m.floorLabel}
function placeName(value){if(!value)return'未設定';const p=splitRef(value),m=floorMaps[p.map];if(p.map==='services')return serviceDestinations[p.id]?.name||'未設定';return p.id==='__plot'?'図面で指定した出発地':m.places.find(x=>x[0]===p.id)?.[1]||'登録通路'}
function placeLabel(value){if(!value)return'未設定';const p=splitRef(value);if(p.map==='services'){const d=serviceDestinations[p.id];return d.building+' '+d.floor+'階｜'+d.name}return mapName(p.map)+'｜'+placeName(value)}
let viewMap='hospital:3',selectedStart=ref('hospital:3','generalReception'),selectedGoal='',route=[],routeGraph=null,step=-1,zoom=1,pan=[0,0],drag=null,plotting=false,crosshair=[500,500],activeSearch=-1,searchMatches=[],wardFilter='全病棟';
const plots={};
let audience='patient',allowUnknown=false,displayMode='clean',editingAccess=false,selectedEdge=null;
let showGeometry=false;
let focusedAmenity=null;
let pendingMapFit=true,pendingViewAction=null;
let accessOverrides={};try{accessOverrides=JSON.parse(localStorage.getItem('maji-access-v2')||'{}')}catch{}
const edgeKey=(a,b)=>[a,b].sort().join('|');
function edgeAccess(map,a,b){const m=floorMaps[map],p=m.nodes[a],q=m.nodes[b];if(m.accessLocked)return m.access?.[edgeKey(a,b)]||'unknown';return accessOverrides[map]?.[edgeKey(a,b)]||m.access?.[edgeKey(a,b)]||'unknown'}
function permitted(access){return access==='patient'||access==='staff'&&audience==='staff'||access==='unknown'&&allowUnknown}

function currentMap(){return floorMaps[viewMap]}
function insideArea(point,polygon){let hit=false;for(let i=0,j=polygon.length-1;i<polygon.length;j=i++){const a=polygon[i],b=polygon[j];if((a[1]>point[1])!==(b[1]>point[1])&&point[0]<(b[0]-a[0])*(point[1]-a[1])/(b[1]-a[1])+a[0])hit=!hit}return hit}
function walkableAreas(m){return m.regions.filter(r=>permitted(r.access)&&r.kind!=='room')}
function patientPointAllowed(m,p){return walkableAreas(m).some(r=>insideArea(p,r.polygon))&&!(m.solidAreas||[]).some(area=>insideArea(p,area))}
function segmentInArea(p,q,polygon,m){const count=Math.max(2,Math.ceil(Math.hypot(p[0]-q[0],p[1]-q[1])*2));for(let i=0;i<=count;i++){const t=(i+.001)/(count+.002);const sample=[p[0]+(q[0]-p[0])*t,p[1]+(q[1]-p[1])*t];if(!insideArea(sample,polygon)||(m.solidAreas||[]).some(area=>insideArea(sample,area)))return false}return true}
function visibleCorridorNodes(m,point){const areas=walkableAreas(m).filter(r=>insideArea(point,r.polygon));if(!areas.length||!patientPointAllowed(m,point))return[];const visible=[];for(const [id,q]of Object.entries(m.nodes)){if(!areas.some(r=>insideArea(q,r.polygon)&&segmentInArea(point,q,r.polygon,m)))continue;const connected=m.edges.some(([a,b])=>(a===id||b===id)&&permitted(edgeAccess(m.key,a,b)));if(!connected)continue;visible.push({node:id,at:q,distance:Math.hypot((point[0]-q[0])*m.ratio,point[1]-q[1])})}return visible.sort((a,b)=>a.distance-b.distance)}
function projectToCorridor(m,point){return visibleCorridorNodes(m,point)[0]||null}
function buildNetwork(){const graph={nodes:{},edges:[],adj:{}};
const add=(u,v,cost,kind,access,path)=>{if(!permitted(access))return;const edge={u,v,cost,kind,access,path};graph.edges.push(edge);(graph.adj[u]??=[]).push({to:v,...edge});(graph.adj[v]??=[]).push({to:u,...edge})};
for(const [key,m]of Object.entries(floorMaps)){
 for(const [id,point]of Object.entries(m.nodes))graph.nodes[ref(key,id)]={map:key,id,point,ratio:m.ratio};
 for(const [a,b]of m.edges){const ek=edgeKey(a,b),access=edgeAccess(key,a,b);const evidence=m.edgeEvidence?.[ek];if(!(evidence?.review==='drawing'||evidence?.review==='user-annotated'&&evidence.confirmedByUser===true)||!m.edgePaths[ek])continue;let path=m.edgePaths[ek];if(Math.hypot(path[0][0]-m.nodes[a][0],path[0][1]-m.nodes[a][1])>.1)path=[...path].reverse();const cost=path.slice(1).reduce((sum,p,i)=>sum+Math.hypot((p[0]-path[i][0])*m.ratio,p[1]-path[i][1]),0);add(ref(key,a),ref(key,b),cost,'walk',access,path)}
 if(plots[key]){graph.nodes[ref(key,'__plot')]={map:key,id:'__plot',point:plots[key],ratio:m.ratio};for(const near of visibleCorridorNodes(m,plots[key]))add(ref(key,'__plot'),ref(key,near.node),near.distance,'walk',audience==='staff'?'staff':'patient',[plots[key],near.at]);}
}
const served=Object.values(floorMaps).filter(m=>m.buildingKey==='hospital').sort((a,b)=>a.floor-b.floor);for(let n=1;n<=6;n++)for(let i=0;i<served.length;i++)for(let j=i+1;j<served.length;j++){const a=served[i],b=served[j];if(a.nodes['ev'+n]&&b.nodes['ev'+n])add(ref(a.key,'ev'+n),ref(b.key,'ev'+n),600*Math.abs(b.floor-a.floor)+300,'elevator','patient',null);}
for(const [key,m] of Object.entries(floorMaps))for(const e of m.viewConnections||[])add(ref(key,e.from),ref(e.toMap,e.to),1,'view','patient',null);const clinics=Object.values(floorMaps).filter(m=>m.buildingKey==='outpatient').sort((a,b)=>a.floor-b.floor);for(let i=0;i<clinics.length;i++)for(let j=i+1;j<clinics.length;j++){const a=clinics[i],b=clinics[j];if(travelMode==='elevator'||!a.nodes.esc||!b.nodes.esc)add(ref(a.key,'ev'),ref(b.key,'ev'),500*Math.abs(b.floor-a.floor)+250,'elevator','patient',null);if(travelMode==='escalator'&&b.floor===a.floor+1&&a.nodes.esc&&b.nodes.esc)add(ref(a.key,'esc'),ref(b.key,'esc'),350,'escalator','patient',null);}return typeof attachConfirmedNavigation==='function'?attachConfirmedNavigation(graph):graph}
function shortest(graph,start,goal){if(!graph.nodes[start]||!graph.nodes[goal])return[];const upperWardTransfer=[start,goal].some(r=>r.startsWith('hospital:')&&floorMaps[graph.nodes[r]?.map]?.floor>=7)&&![start,goal].some(r=>[5,6].includes(floorMaps[graph.nodes[r]?.map]?.floor));const preferFewRides=travelMode==='auto'||travelMode==='elevator';const dist={},rides={},prev={},pending=new Set(Object.keys(graph.nodes));for(const k of pending){dist[k]=Infinity;rides[k]=Infinity;}dist[start]=0;rides[start]=0;while(pending.size){const u=[...pending].reduce((a,b)=>(preferFewRides?(rides[a]<rides[b]||rides[a]===rides[b]&&dist[a]<dist[b]-1e-7):(dist[a]<dist[b]-1e-7||Math.abs(dist[a]-dist[b])<1e-7&&rides[a]<rides[b]))?a:b);if(!Number.isFinite(dist[u]))break;pending.delete(u);if(u===goal)break;for(const edge of graph.adj[u]||[]){if(upperWardTransfer&&['walk','view'].includes(edge.kind)&&[5,6].includes(floorMaps[graph.nodes[u]?.map]?.floor))continue;if(!pending.has(edge.to))continue;const next=dist[u]+edge.cost,nextRides=rides[u]+(['elevator','escalator'].includes(edge.kind)?1:0);if(preferFewRides?(nextRides<rides[edge.to]||nextRides===rides[edge.to]&&next<dist[edge.to]-1e-7):(next<dist[edge.to]-1e-7||Math.abs(next-dist[edge.to])<1e-7&&nextRides<rides[edge.to])){dist[edge.to]=next;rides[edge.to]=nextRides;prev[edge.to]=u}}}if(!Number.isFinite(dist[goal]))return[];const path=[goal];while(path[0]!==start){if(!prev[path[0]])return[];path.unshift(prev[path[0]])}return path}
function transition(from,to){return routeGraph?.adj[from]?.find(e=>e.to===to)?.kind||'walk'}
