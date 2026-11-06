// Register ward EV stops at PUBLIC hall coordinates, outside the gray staff area.
for(const f of [2,4,5,6])if(!floorMaps['hospital:'+f])floorMaps['hospital:'+f]={key:'hospital:'+f,buildingKey:'hospital',title:'病棟',floor:f,floorLabel:f+'F',ratio:1758/1060,nodes:Object.fromEntries([1,2,3,4,5,6].filter(n=>mobilityData.points['hospital:'+f+'::ev'+n]).map(n=>['ev'+n,mobilityData.points['hospital:'+f+'::ev'+n]])),edges:[],places:Object.entries(f===5?mobilityData.unitEntrances:f===6?mobilityData.wardEntrances:{}).map(([id,d])=>[id,d.label]),access:{},edgePaths:{},edgeEvidence:{},regions:[],solidAreas:[],reviewBasis:{rasterSize:[1758,1060]}};
// The newly plotted northern 3P entrance replaces the former area-centre arrival.
mobilityData.points['outpatient:3::3P']=[918.3,331];
mobilityData.portals=mobilityData.portals.filter(p=>p.id!=='outpatient:3::3P');
mobilityData.portals.push({id:'outpatient:3::3P',floor:3,point:[918.3,331],label:'3P 患者支援センターの出入口',arrivalOnly:true});
function attachPublicMobility(graph){
 const replace=new Set(mobilityData.replacedEdges.map(([u,v])=>edgeKey(u,v)));
 graph.edges=graph.edges.filter(e=>!replace.has(edgeKey(e.u,e.v))&&!(['elevator','escalator'].includes(e.kind)&&e.u.startsWith('outpatient:')&&e.v.startsWith('outpatient:')));
 for(const [id,n]of Object.entries(mobilityData.nodes))graph.nodes[id]={...n,ratio:floorMaps[n.map].ratio};
 for(const [u,v,cost]of mobilityData.edges)graph.edges.push({u,v,cost,kind:'walk',access:'patient',path:null,evidence:'official-map-public-mobility'});
 for(const [u,v,cost,kind]of mobilityData.vertical){if(kind==='escalator'&&travelMode==='elevator')continue;graph.edges.push({u,v,cost:cost+(kind==='elevator'&&travelMode==='escalator'?1800:0),kind,access:'patient',path:null});}
 // User confirmed 2026-11-06: the corridor north of patient support is public.
 const supportNorth='outpatient:3::supportNorth';graph.nodes[supportNorth]={map:'outpatient:3',floor:3,point:[911.2,317],label:'患者支援センター北側通路',ratio:floorMaps['outpatient:3'].ratio};
 for(const ref of ['hospital:3::confirmed15','hospital:3::confirmed16']){const n=graph.nodes[ref];if(n)graph.edges.push({u:supportNorth,v:ref,cost:Math.abs(n.point[0]-911.2)+Math.abs(n.point[1]-317),kind:'walk',access:'patient',path:null,evidence:'user-confirmed-2026-11-06-support-north'});}
 // User-plotted 3P-only interior approach, 2026-11-06. Never a public shortcut.
 const supportRef='outpatient:3::3P',only=[supportRef],inside=p=>p&&p[0]>=910&&p[0]<=920&&p[1]>=331&&p[1]<=737;
 graph.edges=graph.edges.filter(e=>e.u!==supportRef&&e.v!==supportRef);
 for(const e of graph.edges)if(floorMaps[graph.nodes[e.u]?.map]?.floor===3&&floorMaps[graph.nodes[e.v]?.map]?.floor===3&&inside(graph.nodes[e.u]?.point)&&inside(graph.nodes[e.v]?.point))e.endpointOnly=only;
 graph.nodes[supportRef]={...graph.nodes[supportRef],point:[918.3,331],displayPoint:[918.3,331]};
 const branch=(label,from,points)=>{let prev=from;for(const [i,p]of points.entries()){const id=i===points.length-1?supportRef:'outpatient:3::supportBranch'+label+i;if(id!==supportRef)graph.nodes[id]={map:'outpatient:3',floor:3,point:p,displayPoint:p,label:'3P専用通路',ratio:floorMaps['outpatient:3'].ratio};const a=graph.nodes[prev].point;graph.edges.push({u:prev,v:id,cost:Math.abs(a[0]-p[0])+Math.abs(a[1]-p[1]),kind:'walk',access:'patient',path:null,endpointOnly:only,evidence:'user-plotted-2026-11-06-3p-only'});prev=id;}};
 branch('North',supportNorth,[[918.3,317],[918.3,331]]);
 branch('South','hospital:3::confirmed36',[[913.9,734.7],[913.9,736.5],[913.9,330.8],[918.3,330.8],[918.3,331]]);
 // Distance-first: identical floor-change units for both modes and all lift banks.
 // Remove historical boarding penalties / escalator discounts from the auto comparison.
 for(const e of graph.edges)if(['elevator','escalator'].includes(e.kind)){const delta=Math.abs(floorMaps[graph.nodes[e.u].map].floor-floorMaps[graph.nodes[e.v].map].floor);e.cost=500*delta+(travelMode==='escalator'&&e.kind==='elevator'?1800:0);}
 graph.adj={};for(const e of graph.edges){(graph.adj[e.u]??=[]).push({to:e.v,...e});(graph.adj[e.v]??=[]).push({to:e.u,...e});}return graph;
}
function closestPublicToilet(graph,from,floor){const all=mobilityData.toilets.map(r=>destinationList.find(d=>d.ref===r)).filter(Boolean),local=all.filter(d=>d.floor===floor);let best=null;for(const d of all){const path=shortest(graph,from,d.ref);if(!path.length)continue;let cost=0;for(let i=1;i<path.length;i++)cost+=graph.adj[path[i-1]].find(e=>e.to===path[i]).cost;if(!best||cost<best.cost)best={destination:d,cost};}return best?.destination||null;}

// Compare actual permitted graph routes, including lift/escalator mode and start entrance options.
function closestPublicEquipment(graph,from,category){let best=null;for(const d of mobilityData.equipment.filter(d=>d.category===category&&d.routeReady&&d.nearestEligible!==false)){const starts=typeof additionalNavigation!=='undefined'&&additionalNavigation.arrivalOptions?.[from]||[from];for(const start of starts){const path=shortest(graph,start,d.ref);if(!path.length)continue;let cost=0;for(let i=1;i<path.length;i++)cost+=graph.adj[path[i-1]].find(e=>e.to===path[i]).cost;if(!best||cost<best.cost)best={destination:d,cost};}}return best?.destination||null;}
