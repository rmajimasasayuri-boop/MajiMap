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
 const sorted=[...maps].sort((a,b)=>a.floor-b.floor),levels={};let y=0;
 sorted.forEach((m,i)=>{if(i)y+=Math.max(sorted[i-1].size[1],m.size[1])*.012+1.2;levels[m.floor]=y;});return levels;
}
function createMap3DRegistration(maps,transfers=[]){
 const offsets={};if(!maps.length)return offsets;offsets[maps[0].floor]=[0,0];
 const raw=e=>[e.point[0],e.point[1]+(e.floor===3?153:0)];
 for(let pass=0;pass<maps.length;pass++)for(const t of transfers.filter(t=>t.kind==='elevator')){
  const a=raw(t.from),b=raw(t.to),oa=offsets[t.from.floor],ob=offsets[t.to.floor];
  if(oa&&!ob)offsets[t.to.floor]=[a[0]+oa[0]-b[0],a[1]+oa[1]-b[1]];
  if(ob&&!oa)offsets[t.from.floor]=[b[0]+ob[0]-a[0],b[1]+ob[1]-a[1]];
 }
 for(const m of maps)offsets[m.floor]??=[0,0];return offsets;
}
function createMap3DDepths(maps){return Object.fromEntries(maps.map(m=>[m.floor,0]));}
function map3DHomeOffset(size,aspect) {
  const radius=Math.max(size[0]/Math.min(aspect,1.6),size[2],size[1]*1.5,10);
  return [0,radius*.94,radius*.86];
}
