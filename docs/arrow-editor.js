// Entrance signs are display annotations. They never modify verified route portals.
const arrowStorageKey='maji-map-entrance-arrows-v1';
const entranceArrowDefaults={};
for(const f of [7,8,9,10])for(const [id,d]of Object.entries(unifiedMapModel.floors[f].wardEntrances))if(d.arrow)entranceArrowDefaults['hospital:'+f+'::'+id]={point:[...d.arrow.point],angle:d.arrow.angle,label:d.label,floor:f};
const arrowEditor={active:false,selected:null,changes:{},undo:[]};
function validateArrowChanges(value){
 if(!value||typeof value!=='object'||Array.isArray(value))throw Error('矢印の修正データを選んでください。');
 const result={};
 for(const [id,a]of Object.entries(value)){
  if(!entranceArrowDefaults[id]||!a||!Array.isArray(a.point)||a.point.length!==2||!a.point.every(Number.isFinite)||a.point[0]<0||a.point[0]>1758||a.point[1]<0||a.point[1]>1060||![0,90,180,270].includes(a.angle))throw Error('位置・向きの形式が合わないため読み込めません。');
  result[id]={point:[...a.point],angle:a.angle};
 }
 return result;
}
try{const stored=localStorage.getItem(arrowStorageKey);if(stored)arrowEditor.changes=validateArrowChanges(JSON.parse(stored));}catch{}
function entranceArrow(id){return arrowEditor.changes[id]||entranceArrowDefaults[id];}
function pushArrowUndo(){arrowEditor.undo.push(JSON.stringify(arrowEditor.changes));if(arrowEditor.undo.length>60)arrowEditor.undo.shift();}
function saveEntranceArrows(){try{localStorage.setItem(arrowStorageKey,JSON.stringify(arrowEditor.changes));ui('arrowSaveStatus').textContent='この端末に保存しました';}catch{ui('arrowSaveStatus').textContent='端末に保存できません。修正データを書き出してください。';}ui('undoArrow').disabled=!arrowEditor.undo.length;}
function setArrowPoint(id,point,save=true){if(!entranceArrowDefaults[id])return;const [w,h]=floorInfo(entranceArrowDefaults[id].floor).size;arrowEditor.changes[id]={point:[Math.max(0,Math.min(w,Math.round(point[0]*10)/10)),Math.max(0,Math.min(h,Math.round(point[1]*10)/10))],angle:entranceArrow(id).angle};if(save)saveEntranceArrows();renderOverlay();}
function selectEntranceArrow(id){if(!entranceArrowDefaults[id])return;arrowEditor.selected=id;syncArrowEditor();renderOverlay();}
function syncArrowEditor(){
 const available=Object.entries(entranceArrowDefaults).filter(([,d])=>d.floor===mapUI.floor);
 document.querySelector('main').classList.toggle('arrow-editing',arrowEditor.active&&available.length>0);
 ui('toggleArrowEditor').hidden=!available.length;ui('toggleArrowEditor').setAttribute('aria-expanded',String(arrowEditor.active));ui('toggleArrowEditor').textContent=arrowEditor.active?'矢印の編集を終了':'入口矢印を編集';
 ui('arrowEditorPanel').hidden=!arrowEditor.active||!available.length;
 if(!available.some(([id])=>id===arrowEditor.selected))arrowEditor.selected=available[0]?.[0]||null;
 ui('arrowSelect').replaceChildren();for(const [id,d]of available){const o=document.createElement('option');o.value=id;o.textContent=d.label;ui('arrowSelect').append(o);}ui('arrowSelect').value=arrowEditor.selected||'';
 if(arrowEditor.selected)ui('arrowDirection').value=entranceArrow(arrowEditor.selected).angle;ui('undoArrow').disabled=!arrowEditor.undo.length;
 ui('walkingHint').hidden=arrowEditor.active;
}
function renderEntranceArrows(svg){
 const s=mapUI.base*mapUI.zoom;
 for(const [id,d]of Object.entries(entranceArrowDefaults)){
  if(d.floor!==mapUI.floor)continue;const a=entranceArrow(id),[x,y]=a.point;
  const g=svgNode('g',{transform:`translate(${x} ${y})`,'data-entrance-ref':id});
  g.append(svgNode('polygon',{points:'11,0 -11,-10 -11,10',transform:`rotate(${a.angle})`,fill:arrowEditor.active&&id===arrowEditor.selected?'#1683d1':'#444444'}));
  if(arrowEditor.active){g.setAttribute('data-arrow-ref',id);g.setAttribute('role','button');g.setAttribute('tabindex','0');g.setAttribute('class','entrance-arrow-handle');g.setAttribute('aria-label',d.label+'の矢印を移動');g.append(svgNode('circle',{cx:0,cy:0,r:15/s,fill:'#1683d11a',stroke:id===arrowEditor.selected?'#1683d1':'#7caecf','stroke-width':2/s,'pointer-events':'all'}));g.onkeydown=e=>{const delta={ArrowUp:[0,-2],ArrowDown:[0,2],ArrowLeft:[-2,0],ArrowRight:[2,0]}[e.key];if(delta){e.preventDefault();e.stopPropagation();selectEntranceArrow(id);pushArrowUndo();setArrowPoint(id,[a.point[0]+delta[0]*(e.shiftKey?5:1),a.point[1]+delta[1]*(e.shiftKey?5:1)]);}if(e.key==='Enter'){e.preventDefault();selectEntranceArrow(id);}};}
  svg.append(g);
 }
}
function initArrowEditor(){
 ui('settings').insertBefore(ui('arrowEditorPanel'),ui('settings').querySelector('.location-card'));
 ui('toggleArrowEditor').onclick=()=>{arrowEditor.active=!arrowEditor.active;mapUI.plot=false;ui('plotCurrent').textContent='地図で出発地を指定';ui('mapPlaceCard').hidden=true;if(arrowEditor.active){document.querySelector('main').classList.remove('settings-hidden');ui('toggleSettings').textContent='設定を閉じる';ui('toggleSettings').setAttribute('aria-expanded','true');stopNavigation();ui('routeSummary').hidden=true;}else if(mapUI.path.length)routePlan();syncArrowEditor();renderOverlay();updateMapMessage();};
 ui('arrowSelect').onchange=()=>selectEntranceArrow(ui('arrowSelect').value);
 ui('arrowDirection').onchange=()=>{const id=arrowEditor.selected;if(!id)return;pushArrowUndo();arrowEditor.changes[id]={point:[...entranceArrow(id).point],angle:Number(ui('arrowDirection').value)};saveEntranceArrows();renderOverlay();};
 document.querySelectorAll('[data-arrow-nudge]').forEach(b=>b.onclick=()=>{const id=arrowEditor.selected;if(!id)return;const [dx,dy]=b.dataset.arrowNudge.split(',').map(Number),a=entranceArrow(id);pushArrowUndo();setArrowPoint(id,[a.point[0]+dx,a.point[1]+dy]);});
 ui('undoArrow').onclick=()=>{const previous=arrowEditor.undo.pop();if(previous===undefined)return;arrowEditor.changes=JSON.parse(previous);saveEntranceArrows();syncArrowEditor();renderOverlay();};
 ui('resetArrow').onclick=()=>{if(!arrowEditor.selected)return;pushArrowUndo();delete arrowEditor.changes[arrowEditor.selected];saveEntranceArrows();syncArrowEditor();renderOverlay();};
 ui('exportArrows').onclick=()=>{const changes={};for(const id of Object.keys(entranceArrowDefaults)){const a=entranceArrow(id);changes[id]={point:[...a.point],angle:a.angle};}const payload={version:1,projectId:'appgprj_6ac5d8a030208191b5a5e6bed09f8cd3',coordinateSystem:'official-map-pixels',changes};const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='Maji-Map-入口矢印.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);ui('arrowSaveStatus').textContent='修正ファイルを書き出しました';};
 ui('importArrows').onclick=()=>ui('arrowImportFile').click();ui('arrowImportFile').onchange=async()=>{const file=ui('arrowImportFile').files[0];if(!file)return;try{const payload=JSON.parse(await file.text());if(payload.version!==1||payload.projectId!=='appgprj_6ac5d8a030208191b5a5e6bed09f8cd3'||payload.coordinateSystem!=='official-map-pixels')throw Error('このマップの矢印データではありません。');const changes=validateArrowChanges(payload.changes);pushArrowUndo();arrowEditor.changes=changes;saveEntranceArrows();syncArrowEditor();renderOverlay();ui('arrowSaveStatus').textContent='修正データを読み込みました';}catch(e){ui('arrowSaveStatus').textContent=e.message;}finally{ui('arrowImportFile').value='';}};
}
