function confirmedArrival(reference){return confirmedNavigation.points[reference]||null;}
function confirmedAnchor(reference){return confirmedNavigation.anchors[reference]||null;}
function attachConfirmedNavigation(graph){
 const floorOf=id=>floorMaps[graph.nodes[id]?.map]?.floor;
 graph.edges=graph.edges.filter(e=>!((e.kind==='walk'||e.kind==='view')&&[1,3].includes(floorOf(e.u))&&floorOf(e.u)===floorOf(e.v)));
 for(const [id,n]of Object.entries(confirmedNavigation.nodes))graph.nodes[id]={...n,ratio:floorMaps[n.map].ratio};
 for(const [u,v,cost]of confirmedNavigation.edges)graph.edges.push({u,v,cost,kind:'walk',access:'patient',path:null,evidence:'user-confirmed-2026-10-22'});
 if(typeof additionalNavigation!=='undefined'){
 const floors=new Set(additionalNavigation.floors);
 graph.edges=graph.edges.filter(e=>!(['walk','view'].includes(e.kind)&&floors.has(floorOf(e.u))&&floorOf(e.u)===floorOf(e.v)&&!(floorOf(e.u)===4&&/::a4/.test(e.u+' '+e.v))));
 for(const [id,n]of Object.entries(additionalNavigation.nodes))graph.nodes[id]={...n,ratio:floorMaps[n.map].ratio};
 for(const [u,v,cost]of additionalNavigation.edges)graph.edges.push({u,v,cost,kind:'walk',access:'patient',path:null,evidence:'official-map-trace-2026-10-23'});
 }
 graph.adj={};for(const edge of graph.edges){(graph.adj[edge.u]??=[]).push({to:edge.v,...edge});(graph.adj[edge.v]??=[]).push({to:edge.u,...edge});}return typeof attachPublicMobility==='function'?attachPublicMobility(graph):graph;
}
