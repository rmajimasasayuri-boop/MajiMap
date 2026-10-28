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

// Arc-length sampling follows each straight segment, including sharp turns.
function createRouteMotionTrack(points) {
  const segments=[];let length=0;
  for(let i=1;i<points.length;i++) {
    const a=points[i-1],b=points[i],size=Math.hypot(...b.map((v,j)=>v-a[j]));
    if(size<.00001)continue;
    segments.push({a,b,start:length,length:size});length+=size;
  }
  return {segments,length};
}
function sampleRouteMotion(track,distance) {
  if(!track.length)return null;
  const at=((distance%track.length)+track.length)%track.length;
  const s=track.segments.find(s=>at<s.start+s.length)||track.segments.at(-1);
  const t=(at-s.start)/s.length;
  return {point:s.a.map((v,j)=>v+(s.b[j]-v)*t),direction:s.b.map((v,j)=>(v-s.a[j])/s.length)};
}
function findMap3DPlace(places,floor,point,boundsAt) {
  return places.find(d=>d.floor===floor&&boundsAt(d).some(([x,y,w,h])=>point[0]>=x&&point[0]<=x+w&&point[1]>=y&&point[1]<=y+h))||null;
}

// Display spacing is based on drawing extents, not real storey heights.
// All floor geometry, pins, transfer lines and hit testing share this layout.
function createMap3DLayout(maps) {
 const levels={};[...maps].sort((a,b)=>a.floor-b.floor).forEach((m,i)=>levels[m.floor]=i*4);return levels;
}
function createMap3DDepths(maps){
 const sorted=[...maps].sort((a,b)=>a.floor-b.floor),depths={};let z=0;
 sorted.forEach((m,i)=>{if(i){const compact=sorted[i-1].floor===2&&m.floor===3;z-=Math.max(sorted[i-1].size[1],m.size[1])*(compact?.009:.01)+(compact?.8:1);}depths[m.floor]=z;});return depths;
}
function map3DHomeOffset(size,aspect) {
  const radius=Math.max(size[0]/Math.min(aspect,1.6),size[2],size[1]*1.5,10);
  return [-radius*.2,radius*.94,radius*.86];
}
