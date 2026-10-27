// The 3D view consumes the existing route; it never computes another route.
function createMap3DData({model, floor, scope, graph, path, nodeFloor, pointAt, linesAt, start, goal}) {
  const routeFloors = graph ? [...new Set(path.map(id => nodeFloor(graph.nodes[id])).filter(Boolean))] : [];
  const floors = scope === 'route' && routeFloors.length ? [...new Set([...routeFloors, floor])].sort((a,b)=>a-b) : [floor];
  const maps = floors.map(f => ({floor:f, ...model.floors[f], offset:[0,f===3?153:0]}));
  const lines = [];
  const transfers = [];
  if(graph && path.length) {
    for(const f of floors) for(const points of linesAt(graph,path,f)) if(points.length>1) lines.push({floor:f,points});
    for(let i=1;i<path.length;i++) {
      const a=path[i-1], b=path[i], edge=graph.adj[a]?.find(e=>e.to===b);
      const fa=nodeFloor(graph.nodes[a]), fb=nodeFloor(graph.nodes[b]);
      if(!['elevator','escalator'].includes(edge?.kind) || !floors.includes(fa) || !floors.includes(fb)) continue;
      const pa=pointAt(a), pb=pointAt(b);
      if(pa&&pb) transfers.push({kind:edge.kind,from:{floor:fa,point:pa},to:{floor:fb,point:pb}});
    }
  }
  return {floor,scope,maps,lines,transfers,start,goal};
}
