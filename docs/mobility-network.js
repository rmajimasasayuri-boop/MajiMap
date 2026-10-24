// Register ward EV stops at PUBLIC hall coordinates, outside the gray staff area.
for(const f of [5,6])if(!floorMaps['hospital:'+f])floorMaps['hospital:'+f]={key:'hospital:'+f,buildingKey:'hospital',title:'病棟',floor:f,floorLabel:f+'F',ratio:1758/1060,nodes:Object.fromEntries([1,2,3,4,5,6].map(n=>['ev'+n,mobilityData.points['hospital:'+f+'::ev'+n]])),edges:[],places:Object.entries(f===5?mobilityData.unitEntrances:mobilityData.wardEntrances).map(([id,d])=>[id,d.label]),access:{},edgePaths:{},edgeEvidence:{},regions:[],solidAreas:[],reviewBasis:{rasterSize:[1758,1060]}};
function attachPublicMobility(graph){
 const replace=new Set(mobilityData.replacedEdges.map(([u,v])=>edgeKey(u,v)));
 graph.edges=graph.edges.filter(e=>!replace.has(edgeKey(e.u,e.v))&&!(['elevator','escalator'].includes(e.kind)&&e.u.startsWith('outpatient:')&&e.v.startsWith('outpatient:')));
 for(const [id,n]of Object.entries(mobilityData.nodes))graph.nodes[id]={...n,ratio:floorMaps[n.map].ratio};
 for(const [u,v,cost]of mobilityData.edges)graph.edges.push({u,v,cost,kind:'walk',access:'patient',path:null,evidence:'official-map-public-mobility'});
 for(const [u,v,cost,kind]of mobilityData.vertical){if(kind==='escalator'&&travelMode==='elevator')continue;graph.edges.push({u,v,cost:cost+(kind==='elevator'&&travelMode==='escalator'?1800:0),kind,access:'patient',path:null});}
 // Distance-first: identical floor-change units for both modes and all lift banks.
 // Remove historical boarding penalties / escalator discounts from the auto comparison.
 for(const e of graph.edges)if(['elevator','escalator'].includes(e.kind)){const delta=Math.abs(floorMaps[graph.nodes[e.u].map].floor-floorMaps[graph.nodes[e.v].map].floor);e.cost=500*delta+(travelMode==='escalator'&&e.kind==='elevator'?1800:0);}
 graph.adj={};for(const e of graph.edges){(graph.adj[e.u]??=[]).push({to:e.v,...e});(graph.adj[e.v]??=[]).push({to:e.u,...e});}return graph;
}
function closestPublicToilet(graph,from,floor){const all=mobilityData.toilets.map(r=>destinationList.find(d=>d.ref===r)).filter(Boolean),local=all.filter(d=>d.floor===floor);let best=null;for(const d of all){const path=shortest(graph,from,d.ref);if(!path.length)continue;let cost=0;for(let i=1;i<path.length;i++)cost+=graph.adj[path[i-1]].find(e=>e.to===path[i]).cost;if(!best||cost<best.cost)best={destination:d,cost};}return best?.destination||null;}

// Compare actual permitted graph routes, including lift/escalator mode and start entrance options.
function closestPublicEquipment(graph,from,category){let best=null;for(const d of mobilityData.equipment.filter(d=>d.category===category&&d.routeReady&&d.nearestEligible!==false)){const starts=typeof additionalNavigation!=='undefined'&&additionalNavigation.arrivalOptions?.[from]||[from];for(const start of starts){const path=shortest(graph,start,d.ref);if(!path.length)continue;let cost=0;for(let i=1;i<path.length;i++)cost+=graph.adj[path[i-1]].find(e=>e.to===path[i]).cost;if(!best||cost<best.cost)best={destination:d,cost};}}return best?.destination||null;}
