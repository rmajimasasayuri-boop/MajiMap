/* Floor-stack presentation inspired by mynrminto/kindai-hospital-navi.
 * Uses its MIT-licensed Three.js r128 / OrbitControls vendor assets.
 * All map geometry and routing remain Maji Map's current confirmed model. */
(() => {
  'use strict';
  if(reviewEmbedded) return;
  const el=id=>document.getElementById(id);
  const host=el('map3D');
  const status=el('map3DStatus');
  const scope='route';
  let mode='2d', interaction='move', library, viewer, pending=0, revision=0;
  const toggle=el('toggleMapView');
  function loadScript(src) {
    return new Promise((resolve,reject)=>{
      const s=document.createElement('script');s.src=src;
      s.onload=resolve;s.onerror=()=>{s.remove();reject(new Error('3D表示のファイルを読み込めませんでした。'));};
      document.head.append(s);
    });
  }
  function libraries() {
    if(!library) library=(async()=>{
      if(!window.THREE) await loadScript('vendor/three.min.js');
      if(!window.THREE.OrbitControls) await loadScript('vendor/OrbitControls.js');
    })().catch(e=>{library=null;throw e;});
    return library;
  }
  function snapshot() {
    return createMap3DData({model:unifiedMapModel,floor:mapUI.floor,scope,
      graph:mapUI.graph,path:mapUI.path,
      nodeFloor:n=>floorMaps[n?.map]?.floor,pointAt:displayPoint,
      linesAt:(g,p,f)=>canDrawFloorRoute(f,p)?floorRouteLines(g,p,f).map(squareRoute):[],
      start:{floor:mapUI.manual?.floor||referenceFloor(mapUI.from),point:mapUI.manual?.point||routeStartPoint()},
      goal:mapUI.goal?{floor:mapUI.goal.floor,point:routeGoalPoint()}:null});
  }
  function queueUpdate() {
    if(mode!=='3d') return;
    cancelAnimationFrame(pending);
    pending=requestAnimationFrame(()=>update());
  }
  async function update() {
    const task=++revision;
    try {
      await libraries();
      if(mode!=='3d'||task!==revision)return;
      if(!viewer) viewer=createViewer(host);
      let data=snapshot();
      if(navigationState.active){
       const step=navigationState.steps[navigationState.index],next=navigationState.steps.slice(navigationState.index+1).find(s=>s.kind==='transfer'),fs=step.kind==='transfer'?[referenceFloor(step.from),referenceFloor(step.to)]:[step.floor,...(next?[referenceFloor(next.to)]:[])];
       data={...data,maps:data.maps.filter(m=>fs.includes(m.floor)),lines:data.lines.filter(l=>fs.includes(l.floor)),transfers:step.kind==='transfer'?data.transfers.filter(t=>fs.includes(t.from.floor)&&fs.includes(t.to.floor)):[]};
      }
      viewer.draw(data);if(navigationState.active&&navigationFollow)viewer.focusStep();
      el('map3DFocusFloors').replaceChildren(...data.maps.map(map=>{const b=document.createElement('button');b.textContent=map.floor+'階を拡大';b.onclick=()=>viewer?.focusFloor(map.floor);return b;}));
      el('map3DFloor').textContent=scope==='route'?'経路全体 · '+data.maps.map(m=>m.floor+'F').join('・'):mapUI.floor+'階';
      status.textContent='矢印の向きに進んでください。病棟名・施設名をクリックして設定できます。';
    } catch(e) {
      if(task!==revision||mode!=='3d')return;
      status.textContent='3Dを表示できません。2Dマップをご利用ください。';
      el('map3DRetry').hidden=false;
    }
  }
  function setMode(next) {
    if(next==='3d'&&(corridorEditor.active||arrowEditor.active))return;
    mode=next;revision++;
    document.body.classList.toggle('view-3d',mode==='3d');
    toggle.textContent=mode==='3d'?'2D':'3D';toggle.setAttribute('aria-label',mode==='3d'?'2Dマップに切り替える':'3Dマップに切り替える');toggle.dataset.currentView=mode;
    el('mapSheet').hidden=mode==='3d';host.hidden=mode!=='3d';
    el('map3DControls').hidden=mode!=='3d';
    el('map3DRetry').hidden=true;
    // Moving between views never changes the selected places or navigation step.
    if(mode==='3d') {status.textContent='3Dマップを準備しています…';queueUpdate();}
    else {cancelAnimationFrame(pending);if(viewer){viewer.dispose();viewer=null;}fitMap();}
  }
  toggle.onclick=()=>setMode(mode==='2d'?'3d':'2d');
  for(const b of document.querySelectorAll('[data-3d-mode]'))b.onclick=()=>{interaction=b.dataset['3dMode'];for(const item of document.querySelectorAll('[data-3d-mode]'))item.setAttribute('aria-pressed',String(item===b));viewer?.setInteraction(interaction);el('map3DGestureHint').textContent=interaction==='move'?'1本指で移動 · 2本指で拡大・回転':'マウスで回転 · 指は1本で移動／2本で拡大・回転';};
  for(const [id,x,y]of [['Left',-1,0],['Right',1,0],['Up',0,1],['Down',0,-1]])el('map3DPan'+id).onclick=()=>viewer?.pan(x,y);
  el('map3DRotateLeft').onclick=()=>viewer?.rotate(-1);
  el('map3DRotateRight').onclick=()=>viewer?.rotate(1);
  el('map3DHome').onclick=()=>viewer?.home();
  el('map3DZoomIn').onclick=()=>viewer?.zoom(.8);
  el('map3DZoomOut').onclick=()=>viewer?.zoom(1.25);
  el('map3DRetry').onclick=()=>{el('map3DRetry').hidden=true;queueUpdate();};
  for(const id of ['annToggle','toggleArrowEditor','editMedicineEVMain'])el(id).addEventListener('click',()=>setMode('2d'),true);
  window.addEventListener('maji-map-update',queueUpdate);window.addEventListener('maji-navigation-focus',()=>viewer?.focusStep());
  new ResizeObserver(()=>viewer?.resize()).observe(host);
  document.addEventListener('visibilitychange',()=>viewer?.pause(document.hidden));
  window.addEventListener('pagehide',()=>viewer?.dispose());
  // Keep the existing 2D pan, click and wheel handlers out of the 3D canvas.
  // OrbitControls r128 listens for mouse move/up on document: allow those to bubble.
  for(const event of ['pointerdown','wheel','keydown'])host.addEventListener(event,e=>e.stopPropagation());
  for(const event of ['pointerdown','wheel','keydown'])el('map3DControls').addEventListener(event,e=>e.stopPropagation());

  function createViewer(container) {
    const T=window.THREE, S=.01;
    const renderer=new T.WebGLRenderer({antialias:true,alpha:false});
    renderer.setPixelRatio(Math.min(devicePixelRatio||1,2));
    renderer.setClearColor(0xeaf0f5,1);renderer.outputEncoding=T.sRGBEncoding;
    renderer.domElement.tabIndex=0;renderer.domElement.setAttribute('aria-label','3Dマップ。1本指で移動、2本指で拡大縮小・回転。');
    container.replaceChildren(renderer.domElement);
    const scene=new T.Scene(), camera=new T.PerspectiveCamera(42,1,.1,500);scene.add(new T.HemisphereLight(0xffffff,0x708799,1));const sunlight=new T.DirectionalLight(0xffffff,.75);sunlight.position.set(-8,22,12);scene.add(sunlight);
    const controls=new T.OrbitControls(camera,renderer.domElement);
    controls.enableDamping=true;controls.maxPolarAngle=Math.PI*.48;
    controls.minDistance=3;controls.maxDistance=100;
    controls.screenSpacePanning=true;controls.enablePan=true;controls.enableRotate=true;
    function setInteraction(value){controls.mouseButtons.LEFT=value==='rotate'?T.MOUSE.ROTATE:T.MOUSE.PAN;controls.mouseButtons.RIGHT=value==='rotate'?T.MOUSE.PAN:T.MOUSE.ROTATE;controls.touches.ONE=T.TOUCH.PAN;controls.touches.TWO=T.TOUCH.DOLLY_ROTATE;}
    setInteraction(interaction);
    const textures=new Map(),loader=new T.TextureLoader(), ray=new T.Raycaster();
    const activePointers=new Set();
    const colorValue=color=>new T.Color(color).convertSRGBToLinear();
    const reducedMotion=window.matchMedia('(prefers-reduced-motion: reduce)');
    let floorLevels={},floorDepths={},floorRegistration={},motion=[],hoverGroup=null,hoverPlace=null,hoverMode=null,progressMarker=null,elevatorCabin=null;
    const hoverCard=el('map3DHover');
    let group=new T.Group(),planes=[],frame,paused=false,disposed=false,previousKey='',homeView,down;
    scene.add(group);
    const offset=f=>(f===3?153:0)+(floorDepths[f]||0)/S;
    const at=(f,p,lift=.1)=>new T.Vector3((p[0]+(floorRegistration[f]?.[0]||0))*S,(floorLevels[f]||0)+lift,(p[1]+offset(f)+(floorRegistration[f]?.[1]||0))*S);
    function freeGroup() {
      group.traverse(o=>{o.geometry?.dispose();for(const m of Array.isArray(o.material)?o.material:o.material?[o.material]:[]){if(m.userData.ownsTexture)m.map?.dispose();m.dispose();}});
      scene.remove(group);group=new T.Group();scene.add(group);planes=[];motion=[];hoverGroup=null;hoverPlace=null;hoverMode=null;hoverCard.hidden=true;progressMarker=null;elevatorCabin=null;
    }
    function texture(src) {
      if(!textures.has(src)) {
        const tex=loader.load(src,()=>{if(!disposed)status.textContent='矢印：進む向き · 施設名をクリックして設定';},undefined,()=>{if(!disposed)status.textContent='一部の地図を読み込めませんでした。2Dへ戻して確認してください。';});
        tex.encoding=T.sRGBEncoding;tex.anisotropy=Math.min(8,renderer.capabilities.getMaxAnisotropy());textures.set(src,tex);
      }
      return textures.get(src);
    }
    function tag(text,point,color,width=2.5) {
      const c=document.createElement('canvas');c.width=512;c.height=96;
      const ctx=c.getContext('2d');ctx.fillStyle=color;ctx.fillRect(0,0,512,96);
      ctx.fillStyle='white';ctx.font='bold 42px sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(text,256,50,480);
      const tex=new T.CanvasTexture(c);tex.encoding=T.sRGBEncoding;
      const mat=new T.SpriteMaterial({map:tex,depthTest:false});mat.userData.ownsTexture=true;
      const sprite=new T.Sprite(mat);sprite.scale.set(width,width*96/512,1);sprite.position.copy(point);sprite.renderOrder=10;group.add(sprite);
    }
    function line(points,color,dashed=false) {
      if(points.length<2)return;
      const material=dashed?new T.LineDashedMaterial({color:colorValue(color),dashSize:.18,gapSize:.12,depthTest:false,transparent:true}):new T.LineBasicMaterial({color:colorValue(color),depthTest:false,transparent:true});
      if(dashed){const l=new T.Line(new T.BufferGeometry().setFromPoints(points),material);l.computeLineDistances();l.renderOrder=8;group.add(l);return;}
      // Straight cylinders preserve each right-angle turn; no smoothing through walls.
      material.dispose();
      for(let i=1;i<points.length;i++) {
        const delta=points[i].clone().sub(points[i-1]),length=delta.length();if(length<.0001)continue;
        const mesh=new T.Mesh(new T.CylinderGeometry(.038,.038,length,8),new T.MeshBasicMaterial({color:colorValue(color),depthTest:false,transparent:true}));
        mesh.position.copy(points[i-1]).addScaledVector(delta,.5);mesh.quaternion.setFromUnitVectors(new T.Vector3(0,1,0),delta.normalize());mesh.renderOrder=8;group.add(mesh);
      }
    }
    function pin(p,label,color,shift=0) {
      if(!p?.point||!planes.some(plane=>plane.userData.floor===p.floor))return;
      const pos=at(p.floor,p.point,.2);pos.x+=shift;
      const ball=new T.Mesh(new T.SphereGeometry(.11,16,12),new T.MeshBasicMaterial({color:colorValue(color),depthTest:false,transparent:true}));ball.position.copy(pos);ball.renderOrder=10;group.add(ball);
      // Pins retain their color without a floating name label.
    }
    function movingArrows(points) {
      const track=createRouteMotionTrack(points.map(p=>p.toArray()));if(!track.length)return;
      const count=Math.min(60,Math.max(1,Math.floor(track.length/.65)));
      const shape=new T.Shape();shape.moveTo(-.11,-.09);shape.lineTo(0,.015);shape.lineTo(.11,-.09);shape.lineTo(.11,-.015);shape.lineTo(0,.10);shape.lineTo(-.11,-.015);shape.closePath();
      for(let i=0;i<count;i++) {
        const arrow=new T.Mesh(new T.ShapeGeometry(shape),new T.MeshBasicMaterial({color:0xffffff,side:T.DoubleSide,depthTest:false,transparent:true}));
        arrow.geometry.rotateX(Math.PI/2);const edge=new T.LineSegments(new T.EdgesGeometry(arrow.geometry),new T.LineBasicMaterial({color:colorValue('#1683d1'),transparent:true,depthTest:false}));edge.renderOrder=10;arrow.add(edge);arrow.renderOrder=9;group.add(arrow);motion.push({arrow,track,offset:i*track.length/count});
      }
    }
    function clearHover() {
      if(hoverGroup){hoverGroup.traverse(o=>{o.geometry?.dispose();o.material?.dispose();});group.remove(hoverGroup);}
      hoverGroup=null;hoverPlace=null;hoverMode=null;container.style.cursor='grab';schedulePhotoRestore();
    }
    function highlight(d,anchor) {
      if(d?.ref===hoverPlace?.ref&&hoverMode===mapUI.pickMode){if(d&&anchor)previewPlacePhoto(d,anchor);return;}
      clearHover();if(!d)return;if(anchor)previewPlacePhoto(d,anchor);hoverPlace=d;hoverMode=mapUI.pickMode;
      const color=mapUI.pickMode==='start'?'#2580c5':'#d83a40';hoverGroup=new T.Group();group.add(hoverGroup);
      outlinePlace(d,color,hoverGroup);
      container.style.cursor='pointer';
    }
    function outlinePlace(d,color,target) {
      if(!planes.some(p=>p.userData.floor===d.floor))return;
      for(const [x,y,w,h]of placeHitBounds(d)) {
        const points=[[x,y],[x+w,y],[x+w,y+h],[x,y+h]].map(p=>at(d.floor,p,.006));
        const fill=new T.Mesh(new T.PlaneGeometry(w*S,h*S),new T.MeshBasicMaterial({color:colorValue(color),side:T.DoubleSide,transparent:true,opacity:.12,depthTest:true,depthWrite:false}));fill.rotation.x=-Math.PI/2;fill.position.copy(at(d.floor,[x+w/2,y+h/2],.004));fill.renderOrder=12;target.add(fill);
        for(let i=0;i<points.length;i++) {
          const a=points[i],b=points[(i+1)%points.length],delta=b.clone().sub(a);
          const edge=new T.Mesh(new T.CylinderGeometry(.009,.009,delta.length(),8),new T.MeshBasicMaterial({color:colorValue(color),depthTest:true,depthWrite:false}));edge.position.copy(a).addScaledVector(delta,.5);edge.quaternion.setFromUnitVectors(new T.Vector3(0,1,0),delta.normalize());edge.renderOrder=13;target.add(edge);
        }
      }
    }
    function pointerHit(e) {
      const r=renderer.domElement.getBoundingClientRect();ray.setFromCamera(new T.Vector2((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1),camera);
      const hit=ray.intersectObjects(planes)[0];if(!hit)return null;
      const floor=hit.object.userData.floor,point=[hit.point.x/S-(floorRegistration[floor]?.[0]||0),hit.point.z/S-offset(floor)-(floorRegistration[floor]?.[1]||0)];
      return {floor,point,place:findMap3DPlace(destinationList,floor,point,placeHitBounds)};
    }
    function draw(data) {
      const key=JSON.stringify([data.floor,data.scope,data.maps.map(m=>m.floor),data.lines,data.transfers,data.start,data.goal,mapUI.goal?.ref,mapUI.from,mapUI.startConfirmed,navigationState.active]);
      if(key===previousKey){if(hoverPlace)highlight(hoverPlace);updateProgress();return;}
      const reset=previousKey!==key;previousKey=key;freeGroup();floorLevels=createMap3DLayout(data.maps);floorDepths=createMap3DDepths(data.maps);floorRegistration=createMap3DRegistration(data.maps,data.transfers);
      const box=new T.Box3();
      for(const map of data.maps) {
        const [w,h]=map.size,active=map.floor===data.floor;
        const mat=new T.MeshBasicMaterial({map:texture(map.image),side:T.DoubleSide,transparent:false,opacity:1});
        const plane=new T.Mesh(new T.PlaneGeometry(w*S,h*S),mat);
        plane.rotation.x=-Math.PI/2;plane.position.copy(at(map.floor,[w/2,h/2],0));
        plane.userData={floor:map.floor,width:w,height:h};planes.push(plane);group.add(plane);
        // Display-only floor plates: all route coordinates remain on the original surface.
        const slab=new T.Mesh(new T.BoxGeometry(w*S,.14,h*S),new T.MeshPhongMaterial({color:colorValue(active?'#c0d8de':'#d4e0e6'),shininess:35}));slab.position.copy(at(map.floor,[w/2,h/2],-.072));group.add(slab);
        const lowerEdge=new T.LineSegments(new T.EdgesGeometry(slab.geometry),new T.LineBasicMaterial({color:colorValue(active?'#2e737d':'#8eaab8'),transparent:true,opacity:.65}));lowerEdge.position.copy(slab.position);group.add(lowerEdge);
        const edge=new T.LineSegments(new T.EdgesGeometry(plane.geometry),new T.LineBasicMaterial({color:active?0x176f68:0x9fb8c4}));edge.rotation.copy(plane.rotation);edge.position.copy(plane.position);group.add(edge);
        tag(map.floor+'F',at(map.floor,[-65,h/2],.3),active?'#176f68':'#607d8c',.9);
        box.expandByPoint(at(map.floor,[0,0],0));box.expandByPoint(at(map.floor,[w,h],0));
      }
      for(const l of data.lines){const points=l.points.map(p=>at(l.floor,p));line(points,0x1683d1);movingArrows(points.map(p=>p.clone().add(new T.Vector3(0,.035,0))));}
      for(const t of data.transfers){const a=at(t.from.floor,t.from.point),b=at(t.to.floor,t.to.point);
       if(t.kind==='elevator'){const height=Math.abs(b.y-a.y),shaft=new T.Mesh(new T.BoxGeometry(.4,height,.4),new T.MeshPhongMaterial({color:colorValue('#67adc9'),transparent:true,opacity:.12,depthWrite:false,shininess:80}));shaft.position.copy(a).lerp(b,.5);group.add(shaft);const frame=new T.LineSegments(new T.EdgesGeometry(shaft.geometry),new T.LineBasicMaterial({color:colorValue('#5b97b0'),transparent:true,opacity:.48}));frame.position.copy(shaft.position);group.add(frame);}
       line([a,b],t.kind==='elevator'?0x07527c:0xd47717,true);movingArrows([a,b]);tag(t.kind==='elevator'?'エレベーター EV':'エスカレーター',a.clone().lerp(b,.5),t.kind==='elevator'?'#07527c':'#ad5a0d',2.3);}
      if(mapUI.startConfirmed){const start=destinationList.find(d=>navigationTargetReference(d)===mapUI.from);if(start)outlinePlace(start,'#2580c5',group);}
      if(mapUI.goal)outlinePlace(mapUI.goal,'#d83a40',group);
      const same=data.start?.floor===data.goal?.floor&&data.start?.point&&data.goal?.point&&Math.hypot(data.start.point[0]-data.goal.point[0],data.start.point[1]-data.goal.point[1])<1;
      if(!navigationState.active)pin(data.start,'出発地','#2580c5',same?-.13:0);pin(data.goal,'目的地','#d83a40',same?.13:0);
      const center=box.getCenter(new T.Vector3()),size=box.getSize(new T.Vector3());
      const cameraOffset=new T.Vector3(...map3DHomeOffset(size.toArray(),camera.aspect));
      // A floor change should frame the new map; pointer selection keeps the view.
      homeView={center,position:center.clone().add(cameraOffset)};
      if(reset)home();resize();updateProgress();
    }
    let trackedPosition=null;
    function updateProgress(){
      const p=navigationState.active&&navigationState.position;
      if(!p||!planes.some(plane=>plane.userData.floor===p.floor)){if(progressMarker)progressMarker.visible=false;if(elevatorCabin)elevatorCabin.visible=false;return;}
      if(!progressMarker){progressMarker=new T.Group();
        const halo=new T.Mesh(new T.SphereGeometry(.19,20,16),new T.MeshBasicMaterial({color:colorValue('#ffffff'),depthTest:false}));
        const dot=new T.Mesh(new T.SphereGeometry(.13,20,16),new T.MeshBasicMaterial({color:colorValue('#2580c5'),depthTest:false}));dot.position.y=.1;halo.renderOrder=19;dot.renderOrder=20;progressMarker.add(halo,dot);group.add(progressMarker);}
      progressMarker.visible=true;if(p.transfer){progressMarker.position.fromArray(transferPosition(p.transfer,end=>at(end.floor,end.point,.12).toArray()));}else progressMarker.position.copy(at(p.floor,p.point,.12));
      if(p.transfer?.kind==='elevator'){if(!elevatorCabin){elevatorCabin=new T.Group();const shell=new T.Mesh(new T.BoxGeometry(.42,.36,.42),new T.MeshPhongMaterial({color:colorValue('#2580c5'),transparent:true,opacity:.7,shininess:80,depthTest:false}));shell.renderOrder=17;const outline=new T.LineSegments(new T.EdgesGeometry(shell.geometry),new T.LineBasicMaterial({color:0xffffff,depthTest:false}));outline.renderOrder=18;elevatorCabin.add(shell,outline);group.add(elevatorCabin);}elevatorCabin.visible=true;elevatorCabin.position.copy(progressMarker.position);}else if(elevatorCabin)elevatorCabin.visible=false;
      if(navigationFollow&&p.transfer&&trackedPosition){const delta=progressMarker.position.clone().sub(trackedPosition);camera.position.add(delta);controls.target.add(delta);controls.update();}trackedPosition=progressMarker.position.clone();
    }
    window.addEventListener('maji-navigation-position',updateProgress);
    function home(){if(!homeView)return;controls.target.copy(homeView.center);camera.position.copy(homeView.position);controls.update();}
    function focusStep(){
      if(!navigationState.active||!navigationFollow)return;
      const step=navigationState.steps[navigationState.index];if(!step)return;
      if(step.kind==='transfer'){
       const p=navigationState.position;if(!p)return;const target=p.transfer?new T.Vector3(...transferPosition(p.transfer,e=>at(e.floor,e.point,.12).toArray())):at(p.floor,p.point,.12);
       controls.target.copy(target);camera.position.copy(target).add(new T.Vector3(-1.5,6,5));trackedPosition=target.clone();controls.update();return;
      }
      const ps=instructionSegments(step).flatMap(s=>s.points).map(p=>at(step.floor,p));if(!ps.length)return;
      const box=new T.Box3().setFromPoints(ps),size=box.getSize(new T.Vector3()),center=box.getCenter(new T.Vector3()),radius=Math.max(size.x/Math.max(camera.aspect,.4),size.z,3.5);
      controls.target.copy(center);camera.position.copy(center).add(new T.Vector3(-radius*.12,radius*1.2,radius*.8));trackedPosition=null;controls.update();clearHover();
    }
    function focusFloor(f){const plane=planes.find(p=>p.userData.floor===f);if(!plane)return;const {width,height}=plane.userData,center=at(f,[width/2,height/2],0),offset=new T.Vector3(...map3DHomeOffset([width*S,0,height*S],camera.aspect));controls.target.copy(center);camera.position.copy(center).add(offset);controls.update();clearHover();}
    function pan(x,y){camera.updateMatrixWorld();const step=camera.position.distanceTo(controls.target)*.09,delta=new T.Vector3().setFromMatrixColumn(camera.matrixWorld,0).multiplyScalar(x*step).addScaledVector(new T.Vector3().setFromMatrixColumn(camera.matrixWorld,1),y*step);camera.position.add(delta);controls.target.add(delta);controls.update();clearHover();}
    function rotate(direction){camera.position.sub(controls.target).applyAxisAngle(new T.Vector3(0,1,0),direction*Math.PI/12).add(controls.target);controls.update();clearHover();}
    function zoom(factor){camera.position.sub(controls.target).multiplyScalar(factor).add(controls.target);controls.update();}
    function resize(){const w=container.clientWidth,h=container.clientHeight;if(!w||!h)return;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();}
    function loop(time=0){if(disposed||paused)return;frame=requestAnimationFrame(loop);for(const m of motion){const sample=sampleRouteMotion(m.track,m.offset+(reducedMotion.matches?0:time*.00065));m.arrow.position.fromArray(sample.point);m.arrow.quaternion.setFromUnitVectors(new T.Vector3(0,0,1),new T.Vector3(...sample.direction));}controls.update();renderer.render(scene,camera);}
    function pause(value){paused=value;cancelAnimationFrame(frame);if(!paused&&!disposed)loop();}
    renderer.domElement.addEventListener('pointerdown',e=>{if(navigationState.active)navigationFollow=false;activePointers.add(e.pointerId);down=activePointers.size===1?{x:e.clientX,y:e.clientY,id:e.pointerId,moved:false}:null;});
    renderer.domElement.addEventListener('pointermove',e=>{if(down){if(Math.hypot(e.clientX-down.x,e.clientY-down.y)>5){down.moved=true;hidePlacePreview();}clearHover();return;}if(activePointers.size)return;highlight(pointerHit(e)?.place,{x:e.clientX,y:e.clientY});});
    renderer.domElement.addEventListener('pointerleave',()=>clearHover());
    renderer.domElement.addEventListener('pointerup',e=>{
      const pressed=down;down=null;activePointers.delete(e.pointerId);if(!pressed||pressed.id!==e.pointerId||pressed.moved||e.button!==0)return;
      const hit=pointerHit(e);if(!hit)return;
      // A facility is selected on the first click, including on another visible floor.
      if(hit.place)previewPlacePhoto(hit.place,{x:e.clientX,y:e.clientY},true);
      else {if(mapUI.floor!==hit.floor)showFloor(hit.floor,false);selectMapPoint(hit.point);}
    });
    renderer.domElement.addEventListener('pointercancel',e=>{activePointers.delete(e.pointerId);down=null;});
    const releasePointer=e=>{activePointers.delete(e.pointerId);down=null;};
    document.addEventListener('pointerup',releasePointer);document.addEventListener('pointercancel',releasePointer);
    resize();loop();
    function dispose(){disposed=true;window.removeEventListener('maji-navigation-position',updateProgress);document.removeEventListener('pointerup',releasePointer);document.removeEventListener('pointercancel',releasePointer);cancelAnimationFrame(frame);controls.dispose();freeGroup();for(const t of textures.values())t.dispose();renderer.dispose();renderer.forceContextLoss();renderer.domElement.remove();}
    return {draw,home,zoom,resize,pause,dispose,pan,rotate,focusFloor,focusStep,setInteraction};
  }
})();
