// Visitor-facing readability, preview cards and one editable waypoint.
function closeVisitorPreview(){mapUI.previewPlace=null;document.body.classList.remove('place-preview-open');ui('facilityDetails').hidden=true;if(typeof renderOverlay==='function')renderOverlay();}
function showVisitorPreview(d){
 mapUI.previewPlace=d;renderPlacePhoto(d);document.body.classList.add('place-preview-open');ui('facilityDetails').hidden=false;updatePreviewActions();renderOverlay();
}
function updatePreviewActions(){
 const d=destinationPhotoState.place,b=ui('detailNavigate');b.textContent='ナビ開始';b.disabled=!d||isGuideOnly(d);
 ui('facilityDetails').dataset.previewAction='navigate';
}
function navigateVisitorPreview(){
 const d=destinationPhotoState.place;if(!d)return;
 startingNavigationFromPreview=true;
 try{
  closeVisitorPreview();choosePlace(d,true);rememberPlace(d);enterRoutePlanning();
  if(ensureIndoorStart()&&navigationState.steps.length){navigationFollow=true;startNavigation();}
 }finally{
  startingNavigationFromPreview=false;syncRoutePanels();
 }
}
function syncVisitorRefinements(){
 updatePreviewActions();
 requestAnimationFrame(layoutVisitorControls);
}
// Rectangular bounds measured from the displayed SVG artwork, with a small
// border for the outermost outlines and symbols. Coordinates remain unchanged.
const visitorArtworkBounds = {
 1:[9,173,1740,856], 2:[20,173,1957,1359], 3:[0,17,1758,1516],
 4:[12,169,1720,855], 5:[0,168,1731,857], 6:[13,169,1719,857],
 7:[884,170,849,855], 8:[884,170,849,855], 9:[884,170,849,855], 10:[884,170,849,855]
};
function visitorCropBounds(floor=mapUI.floor){
 const [x,y,w,h]=visitorArtworkBounds[floor];
 return {x,y,w,h};
}
function visitorMapArea(){
 const viewport=ui('mapViewport'),box=viewport.getBoundingClientRect(),safe=visitorSafeArea(),gap=6;
 const visible=id=>{const el=ui(id),rect=el?.getBoundingClientRect();return rect?.width&&rect.height&&getComputedStyle(el).display!=='none'?rect:null;};
 const rail=document.querySelector('.map-navigation-rail'),railRect=rail?.getBoundingClientRect();
 const view=visible('toggleMapView'),search=visible('placeSearchBar')||visible('findPlace');
 const planner=visible('routePlanner');
 const top=Math.min(viewport.clientHeight*.42,Math.max(safe.top, ...[view,search,planner].filter(Boolean).map(r=>r.bottom-box.top))+gap);
 const panels=['facilityDetails','navigationPanel','routeOptionsDock'].map(visible).filter(Boolean);
 const bottom=Math.max(top+60,Math.min(viewport.clientHeight-safe.bottom,...panels.map(r=>r.top-box.top))-gap);
 const left=safe.left+gap,right=Math.max(left+100,Math.min(viewport.clientWidth-safe.right,railRect?.width?railRect.left-box.left:Infinity)-gap);
 return {left,right,top,bottom};
}
function boundedVisitorPan(pan,size,scale,view,area,bounds={x:0,y:0,w:size[0],h:size[1]}){
 const origin=[view.width/2-size[0]*scale/2+pan[0]+bounds.x*scale,view.height/2-size[1]*scale/2+pan[1]+bounds.y*scale];
 const extent=[bounds.w*scale,bounds.h*scale];
 return pan.map((value,i)=>{
  const first=i?area.top:area.left,last=i?area.bottom:area.right,length=last-first;
  const edge=extent[i]<=length?first+(length-extent[i])/2:Math.max(last-extent[i],Math.min(first,origin[i]));
  return value+edge-origin[i];
 });
}
function visitorCropActive(){
 return !reviewEmbedded&&!corridorEditor.active&&!arrowEditor.active&&!document.body.classList.contains('outdoor-position-editing');
}
function constrainVisitorPan(){
 if(document.body.classList.contains('view-3d')||!visitorCropActive())return mapUI.pan;
 const v=ui('mapViewport');
 return boundedVisitorPan(mapUI.pan,floorInfo(mapUI.floor).size,mapUI.base*mapUI.zoom,
  {width:v.clientWidth,height:v.clientHeight},visitorMapArea(),visitorCropBounds());
}
const visitorFavoritesKey='maji-place-favorites-v1';
function readPlaceFavorites(){try{const list=JSON.parse(localStorage.getItem(visitorFavoritesKey)||'[]');return Array.isArray(list)?[...new Set(list.filter(ref=>typeof ref==='string'&&destinationList.some(d=>d.ref===ref&&d.usage!=='emergency-only')).map(destinationIdentity))].slice(0,10):[];}catch{return [];}}
function updateVisitorFavorites(){
 const list=ui('favoritePlaces');if(!list)return;const refs=readPlaceFavorites();list.replaceChildren();for(const ref of refs){const d=destinationList.find(p=>p.ref===ref),row=document.createElement('div');row.className='favorite-place-row';const select=document.createElement('button');select.textContent=destinationDisplayName(d);select.onclick=()=>selectPlace(d);const remove=document.createElement('button');remove.textContent='☆';remove.setAttribute('aria-label',destinationDisplayName(d)+'のお気に入りを解除');remove.onclick=()=>toggleVisitorFavorite(d);row.append(select,remove);list.append(row);}
 ui('favoriteEmpty').hidden=Boolean(refs.length);const d=destinationPhotoState.place,b=ui('favoritePlaceButton'),saved=d&&refs.includes(destinationIdentity(d.ref));b.textContent=saved?'★':'☆';b.setAttribute('aria-label',saved?'お気に入りを解除':'お気に入りに登録');b.setAttribute('aria-pressed',String(Boolean(saved)));b.disabled=!d||d.type==='manual';b.hidden=d?.usage==='emergency-only';b.title='このブラウザ・サイトに保存します';
}
function toggleVisitorFavorite(d=destinationPhotoState.place){if(!d||d.type==='manual'||d.usage==='emergency-only')return;const id=destinationIdentity(d.ref),refs=readPlaceFavorites(),next=refs.includes(id)?refs.filter(ref=>ref!==id):[id,...refs];if(!refs.includes(id)&&refs.length>=10){ui('favoriteStatus').textContent='お気に入りは10件までです。先に1件解除してください。';ui('favoriteStatus').hidden=false;return;}try{localStorage.setItem(visitorFavoritesKey,JSON.stringify(next));ui('favoriteStatus').textContent='';ui('favoriteStatus').hidden=true;updateVisitorFavorites();}catch{ui('favoritePlaceButton').textContent='保存できませんでした';ui('favoriteStatus').textContent='このブラウザでは保存できません。';ui('favoriteStatus').hidden=false;}}
function initVisitorFavorites(){
 const section=document.createElement('section');section.className='favorite-section';section.innerHTML='<h3>お気に入り</h3><p id="favoriteStatus" role="status" hidden></p><div id="favoritePlaces"></div><p id="favoriteEmpty">施設の写真カードの☆から登録できます。</p>';ui('searchChoices').insertBefore(section,ui('searchChoices').querySelector('.recent-heading'));
 const b=document.createElement('button');b.id='favoritePlaceButton';b.className='favorite-place-button';b.onclick=()=>toggleVisitorFavorite();ui('facilityDetails').querySelector('.facility-detail-body').append(b);updateVisitorFavorites();
}
function visitorSafeArea(){const c=getComputedStyle(document.documentElement);return Object.fromEntries(['top','bottom','left','right'].map(side=>[side,parseFloat(c.getPropertyValue('--visitor-safe-'+side))||0]));}
function visitorRailLayout({height,top,bottom,wide}){
 return {compact:false,height:320};
}
function layoutVisitorControls(){
 const v=ui('mapViewport'),box=v.getBoundingClientRect(),planner=ui('routePlanner'),p=planner.getBoundingClientRect(),planning=document.body.classList.contains('route-planning'),active=navigationState.active,safe=visitorSafeArea(),wide=box.width>box.height;
 const visible=el=>!el.hidden&&getComputedStyle(el).display!=='none'&&el.getBoundingClientRect().height>0;
 const top=8+safe.top;
 const dock=ui('routeOptionsDock');let bottom=visible(dock)?Math.max(40+safe.bottom,box.bottom-dock.getBoundingClientRect().top+12):40+safe.bottom;for(const id of ['facilityDetails','navigationPanel']){const el=ui(id),r=el.getBoundingClientRect();if(visible(el))bottom=Math.max(bottom,box.bottom-r.top+8);}
 const rail=visitorRailLayout({height:box.height,top,bottom,wide});document.body.classList.toggle('compact-floor-rail',rail.compact);
 document.documentElement.style.setProperty('--route-panel-bottom',top+'px');document.documentElement.style.setProperty('--visitor-floor-height',rail.height+'px');
 const floors=ui('allFloors');for(const b of floors.querySelectorAll('[data-floor]'))b.style.setProperty('--floor-number',b.dataset.floor);const activeFloor=floors.querySelector('[data-floor="'+mapUI.floor+'"]');if(activeFloor&&(floors.dataset.visibleFloor!==String(mapUI.floor)||floors.dataset.layoutHeight!==String(rail.height))){floors.dataset.visibleFloor=mapUI.floor;floors.dataset.layoutHeight=rail.height;const f=activeFloor.getBoundingClientRect(),r=floors.getBoundingClientRect();if(f.top<r.top)floors.scrollTop+=f.top-r.top;else if(f.bottom>r.bottom)floors.scrollTop+=f.bottom-r.bottom;}document.documentElement.style.setProperty('--visitor-floor-height',rail.height+'px');
}
function syncVisitorVisualViewport(){
 const viewport=window.visualViewport,height=viewport&&Math.abs(viewport.scale-1)<.02?viewport.height:window.innerHeight;
 document.documentElement.style.setProperty('--visitor-screen-height',Math.round(height)+'px');requestAnimationFrame(layoutVisitorControls);
}
function initTransientMapMessage(){
 const e=ui('mapMessage');if(!e)return;let timer,fade;const show=()=>{clearTimeout(timer);clearTimeout(fade);e.classList.remove('message-faded','message-fading');if(!/この位置は通路接続が未確認です|通路に接続できる場所/.test(e.textContent))return;timer=setTimeout(()=>{e.classList.add('message-fading');fade=setTimeout(()=>e.classList.add('message-faded'),1200);},2500);};new MutationObserver(show).observe(e,{childList:true,characterData:true,subtree:true});
}
function initEquipmentKindField(){
 const field=document.createElement('label');field.id='equipmentKindField';field.className='equipment-kind-field';field.hidden=true;field.textContent='設備の種類';const select=document.createElement('select');select.id='equipmentKind';select.setAttribute('aria-label','設備の種類');const choices=[['','すべての設備'],['toilet','トイレ'],['payment','自動支払機'],['parking','駐車場精算機'],['vending','自動販売機'],['aed','AED'],['phone','公衆電話'],['baby','ベビー設備'],['bloodPressure','血圧計'],['scale','身長・体重計']];for(const [value,name]of choices){const option=document.createElement('option');option.value=value;option.textContent=name;select.append(option);}field.append(select);ui('placeResults').before(field);select.onchange=()=>{mapUI.equipmentKind=select.value;ui('findPlace').value='';mapUI.resultFloor=null;renderResults();requestAnimationFrame(()=>{ui('placeResults').style.paddingBottom='12px';ui('placeResults').scrollTop=0;});};
 for(const b of document.querySelectorAll('[data-filter]'))b.addEventListener('click',()=>{mapUI.equipmentKind='';select.value='';renderResults();});ui('findPlace').addEventListener('input',()=>{mapUI.equipmentKind='';select.value='';});
 const match=resultCategoryMatches;resultCategoryMatches=function(d){if(!match(d))return false;if(mapUI.filter!=='equipment'||!mapUI.equipmentKind)return true;return mapUI.equipmentKind==='toilet'?d.type==='toilet':d.type==='equipment'&&d.category===mapUI.equipmentKind;};
}
function initVisitorRefinements(){
 if(reviewEmbedded)return;
 for(const id of ['routeStartField','routeGoalField'])ui(id).addEventListener('click',()=>{mapUI.addingWaypoint=false;ui('findPlace').placeholder='診療科・病棟・施設を検索';},true);
 ui('openPlaceSearch').addEventListener('click',()=>{if(!document.querySelector('main').classList.contains('settings-hidden'))mapUI.addingWaypoint=false;},true);
 const originalPhoto=renderPlacePhoto;renderPlacePhoto=function(d){originalPhoto(d);updatePreviewActions();updateVisitorFavorites();};
 const originalHide=hidePlacePreview;hidePlacePreview=function(){originalHide();mapUI.previewPlace=null;document.body.classList.remove('place-preview-open');};
 previewPlacePhoto=function(d,anchor,pinned=false){if(!pinned||!d)return;if(mapUI.pickMode==='start'){selectStartPlace(d,true);return;}showVisitorPreview(d);};
 ui('detailRoute').onclick=()=>{const d=destinationPhotoState.place;if(!d)return;closeVisitorPreview();choosePlace(d,true);rememberPlace(d);enterRoutePlanning();};
 ui('detailNavigate').onclick=navigateVisitorPreview;ui('detailWebsite').textContent='ウェブサイト';ui('detailWebsite').setAttribute('aria-label','ウェブサイト');
 ui('closeFacilityDetails').onclick=closeVisitorPreview;
 ui('routeDockNavigate').textContent='ナビ開始';
 const reset=document.createElement('button');reset.id='visitorMapHome';reset.className='floor-home-button';reset.textContent='全体';reset.onclick=()=>document.body.classList.contains('view-3d')?ui('map3DHome').click():ui('resetMap').click();document.querySelector('.map-navigation-rail').append(reset,ui('returnToPosition'));
 const home=ui('map3DHome');home.classList.add('floor-home-button');document.querySelector('.map-navigation-rail').append(home);
 const wheelchair=document.querySelector('[data-travel-choice="elevator"]');wheelchair.setAttribute('aria-label','車椅子（EVのみ）');wheelchair.title='階の移動にはエレベーターのみを使います';for(const node of [...wheelchair.childNodes])if(node.nodeType===3)node.textContent='車椅子';
 // Equipment icons are subfilters shown only inside Equipment, avoiding duplicate categories.
 initEquipmentKindField();
 const originalResults=renderResults;renderResults=function(){originalResults();ui('equipmentKindField').hidden=mapUI.filter!=='equipment';};
 const originalCategory=resultCategoryMatches;resultCategoryMatches=function(d){if(mapUI.filter==='amenity')return ['services::restaurant','services::cafe','services::seven','services::green','services::cosmetics','services::glasses','services::hair'].includes(d.ref);return originalCategory(d);};
 window.addEventListener('maji-map-update',syncVisitorRefinements);const observer=new ResizeObserver(()=>requestAnimationFrame(layoutVisitorControls));for(const id of ['mapViewport','routePlanner','facilityDetails','routeOptionsDock'])observer.observe(ui(id));initTransientMapMessage();window.visualViewport?.addEventListener('resize',syncVisitorVisualViewport);window.addEventListener('resize',syncVisitorVisualViewport);syncVisitorVisualViewport();
 initVisitorFavorites();updatePreviewActions();renderResults();syncVisitorRefinements();applyMapTransform();
}
initVisitorRefinements();
