// Visitor-facing readability, preview cards and one editable waypoint.
function closeVisitorPreview(){mapUI.previewPlace=null;document.body.classList.remove('place-preview-open');ui('facilityDetails').hidden=true;if(typeof renderOverlay==='function')renderOverlay();}
function showVisitorPreview(d){
 mapUI.previewPlace=d;renderPlacePhoto(d);document.body.classList.add('place-preview-open');ui('facilityDetails').hidden=false;updatePreviewActions();renderOverlay();
}
function previewOffersWaypoint(d){return Boolean(mapUI.addingWaypoint&&d&&mapUI.goal&&d.ref!==mapUI.goal.ref);}
function updatePreviewActions(){
 const d=destinationPhotoState.place,b=ui('detailNavigate'),via=previewOffersWaypoint(d);b.textContent=via?'経由地を追加':'ナビ開始';b.disabled=!d||isGuideOnly(d)||(via&&d.type==='manual');b.title=via?'この場所を通る経路に変更します':'';
 ui('facilityDetails').dataset.previewAction=via?'waypoint':'navigate';
}
function setVisitorWaypoint(d){
 if(!d||!mapUI.goal||isGuideOnly(d)||d.ref===mapUI.goal.ref||d.type==='manual')return;
 stopNavigation();mapUI.addingWaypoint=false;mapUI.waypoint=d;rememberPlace(d);closeVisitorPreview();enterRoutePlanning();syncVisitorRefinements();
}
function addVisitorWaypoint(){setVisitorWaypoint(destinationPhotoState.place);}
function beginWaypointSelection(){
 if(!mapUI.goal)return;if(mapUI.addingWaypoint){mapUI.addingWaypoint=false;syncVisitorRefinements();return;}
 closeVisitorPreview();mapUI.addingWaypoint=true;setSelectionMode('goal');togglePlacePanel(true);ui('findPlace').placeholder='経由地を検索';ui('findPlace').focus({preventScroll:true});syncVisitorRefinements();
}
function navigateVisitorPreview(){
 const d=destinationPhotoState.place;if(!d)return;if(previewOffersWaypoint(d)){addVisitorWaypoint();return;}
 closeVisitorPreview();choosePlace(d,true);rememberPlace(d);enterRoutePlanning();if(ensureIndoorStart()&&navigationState.steps.length){navigationFollow=true;startNavigation();syncRoutePanels();}
}
function syncVisitorRefinements(){
 updatePreviewActions();const add=ui('routeAddWaypoint');if(add){add.textContent=mapUI.addingWaypoint?'経由地選択中':'経由地を追加';add.setAttribute('aria-pressed',String(Boolean(mapUI.addingWaypoint)));}
 const row=ui('routeViaRow');if(row){row.hidden=!mapUI.waypoint;if(mapUI.waypoint)ui('routeViaLabel').textContent=destinationDisplayName(mapUI.waypoint);}
 ui('chooseCurrentMapPoint').hidden=true;
 requestAnimationFrame(layoutVisitorControls);
}
function boundedVisitorPan(pan,size,scale,view,area,bounds={x:0,y:0,w:size[0],h:size[1]}){
 const origin=[view.width/2-size[0]*scale/2+pan[0]+bounds.x*scale,view.height/2-size[1]*scale/2+pan[1]+bounds.y*scale],extent=[bounds.w*scale,bounds.h*scale];
 return pan.map((v,i)=>{const first=i?area.top:area.left,last=i?area.bottom:area.right,length=last-first;const next=extent[i]<=length?first+(length-extent[i])/2:Math.max(last-extent[i]-20,Math.min(first+20,origin[i]));return v+next-origin[i];});
}
function constrainVisitorPan(){
 if(document.body.classList.contains('view-3d'))return mapUI.pan;
 const v=ui('mapViewport'),box=v.getBoundingClientRect(),width=v.clientWidth,height=v.clientHeight;const safe=visitorSafeArea();let top=66+safe.top,bottom=height-44-safe.bottom;
 for(const id of ['routePlanner','facilityDetails','navigationPanel']){const e=ui(id),r=e.getBoundingClientRect();if(!r.height||getComputedStyle(e).display==='none')continue;if(id==='routePlanner')top=Math.max(top,r.bottom-box.top+10);else bottom=Math.min(bottom,r.top-box.top-10);}
 top=Math.min(top,height*.45);bottom=Math.max(bottom,top+120);const area={left:12+safe.left,right:Math.max(160,width-76-safe.right),top,bottom:Math.min(height-36,bottom)};
 return boundedVisitorPan(mapUI.pan,floorInfo(mapUI.floor).size,mapUI.base*mapUI.zoom,{width,height},area,mapUI.floor>=7?{x:897,y:183,w:823,h:829}:undefined);
}
const visitorFavoritesKey='maji-place-favorites-v1';
function readPlaceFavorites(){try{const list=JSON.parse(localStorage.getItem(visitorFavoritesKey)||'[]');return Array.isArray(list)?[...new Set(list.filter(ref=>typeof ref==='string'&&destinationList.some(d=>d.ref===ref)))]:[];}catch{return [];}}
function updateVisitorFavorites(){
 const list=ui('favoritePlaces');if(!list)return;const refs=readPlaceFavorites();list.replaceChildren();for(const ref of refs){const d=destinationList.find(p=>p.ref===ref),row=document.createElement('div');row.className='favorite-place-row';const select=document.createElement('button');select.textContent=d.floor+'階 '+destinationDisplayName(d);select.onclick=()=>selectPlace(d);const remove=document.createElement('button');remove.textContent='☆';remove.setAttribute('aria-label',destinationDisplayName(d)+'のお気に入りを解除');remove.onclick=()=>toggleVisitorFavorite(d);row.append(select,remove);list.append(row);}
 ui('favoriteEmpty').hidden=Boolean(refs.length);const d=destinationPhotoState.place,b=ui('favoritePlaceButton'),saved=d&&refs.includes(d.ref);b.textContent=saved?'★':'☆';b.setAttribute('aria-label',saved?'お気に入りを解除':'お気に入りに登録');b.setAttribute('aria-pressed',String(Boolean(saved)));b.disabled=!d||d.type==='manual';b.title='このブラウザ・サイトに保存します';
}
function toggleVisitorFavorite(d=destinationPhotoState.place){if(!d||d.type==='manual')return;const refs=readPlaceFavorites(),next=refs.includes(d.ref)?refs.filter(ref=>ref!==d.ref):[d.ref,...refs];try{localStorage.setItem(visitorFavoritesKey,JSON.stringify(next));ui('favoriteStatus').textContent='';ui('favoriteStatus').hidden=true;updateVisitorFavorites();}catch{ui('favoritePlaceButton').textContent='保存できませんでした';ui('favoriteStatus').textContent='このブラウザでは保存できません。';ui('favoriteStatus').hidden=false;}}
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
 const e=ui('mapMessage');let timer,fade;const show=()=>{clearTimeout(timer);clearTimeout(fade);e.classList.remove('message-faded','message-fading');if(!e.textContent.includes('この位置は通路接続が未確認です'))return;timer=setTimeout(()=>{e.classList.add('message-fading');fade=setTimeout(()=>e.classList.add('message-faded'),1200);},4200);};new MutationObserver(show).observe(e,{childList:true,characterData:true,subtree:true});
}
function initEquipmentKindField(){
 const field=document.createElement('label');field.id='equipmentKindField';field.className='equipment-kind-field';field.hidden=true;field.textContent='設備の種類';const select=document.createElement('select');select.id='equipmentKind';select.setAttribute('aria-label','設備の種類');const choices=[['','すべての設備'],['toilet','トイレ'],['payment','自動支払機'],['parking','駐車場精算機'],['vending','自動販売機'],['aed','AED'],['phone','公衆電話'],['baby','ベビールーム'],['scale','身長・体重計']];for(const [value,name]of choices){const option=document.createElement('option');option.value=value;option.textContent=name;select.append(option);}field.append(select);ui('placeResults').before(field);select.onchange=()=>{mapUI.equipmentKind=select.value;ui('findPlace').value='';mapUI.resultFloor=null;renderResults();requestAnimationFrame(()=>{ui('placeResults').style.paddingBottom='12px';ui('placeResults').scrollTop=0;});};
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
 const select=selectPlace;selectPlace=function(d,...args){if(mapUI.addingWaypoint){setVisitorWaypoint(d);return;}return select(d,...args);};
 ui('detailRoute').onclick=()=>{const d=destinationPhotoState.place;if(!d)return;closeVisitorPreview();choosePlace(d,true);rememberPlace(d);enterRoutePlanning();};
 ui('detailNavigate').onclick=navigateVisitorPreview;ui('detailWebsite').textContent='ウェブサイト';ui('detailWebsite').setAttribute('aria-label','ウェブサイト');
 ui('closeFacilityDetails').onclick=closeVisitorPreview;
 const row=document.createElement('div');row.id='routeViaRow';row.className='route-via-row';row.hidden=true;const label=document.createElement('strong');label.id='routeViaLabel';const copy=document.createElement('button');copy.className='route-via-copy';copy.innerHTML='<small>経由地</small>';copy.append(label);copy.setAttribute('aria-label','経由地を変更');copy.onclick=beginWaypointSelection;const remove=document.createElement('button');remove.textContent='×';remove.setAttribute('aria-label','経由地を削除');remove.onclick=()=>{mapUI.waypoint=null;routePlan();syncVisitorRefinements();};row.append(copy,remove);ui('routePlanner').querySelector('.route-place-fields').insertBefore(row,ui('routeGoalField'));
 const viaButton=document.createElement('button');viaButton.id='routeAddWaypoint';viaButton.textContent='経由地を追加';viaButton.onclick=beginWaypointSelection;ui('routeOptionsDock').prepend(viaButton);
 ui('routeDockNavigate').textContent='ナビ開始';
 const reset=document.createElement('button');reset.id='visitorMapHome';reset.className='floor-home-button';reset.textContent='全体';reset.onclick=()=>document.body.classList.contains('view-3d')?ui('map3DHome').click():ui('resetMap').click();document.querySelector('.map-navigation-rail').append(reset);
 const home=ui('map3DHome');home.classList.add('floor-home-button');document.querySelector('.map-navigation-rail').append(home);
 const wheelchair=document.querySelector('[data-travel-choice="elevator"]');wheelchair.setAttribute('aria-label','車椅子（EVのみ）');wheelchair.title='階の移動にはエレベーターのみを使います';for(const node of [...wheelchair.childNodes])if(node.nodeType===3)node.textContent='車椅子';
 // Equipment icons are subfilters shown only inside Equipment, avoiding duplicate categories.
 const equipment=ui('nearbyEquipmentBar');equipment.classList.add('equipment-subfilters');equipment.setAttribute('aria-label','設備の種類');
 initEquipmentKindField();
 const originalResults=renderResults;renderResults=function(){originalResults();equipment.hidden=true;ui('equipmentKindField').hidden=mapUI.filter!=='equipment';};
 const originalCategory=resultCategoryMatches;resultCategoryMatches=function(d){if(mapUI.filter==='amenity')return ['services::restaurant','services::cafe','services::seven','services::green','services::cosmetics','services::glasses','services::hair'].includes(d.ref);return originalCategory(d);};
 window.addEventListener('maji-map-update',syncVisitorRefinements);const observer=new ResizeObserver(()=>requestAnimationFrame(layoutVisitorControls));for(const id of ['mapViewport','routePlanner','facilityDetails','routeOptionsDock'])observer.observe(ui(id));initTransientMapMessage();window.visualViewport?.addEventListener('resize',syncVisitorVisualViewport);window.addEventListener('resize',syncVisitorVisualViewport);syncVisitorVisualViewport();
 initVisitorFavorites();updatePreviewActions();renderResults();syncVisitorRefinements();applyMapTransform();
}
initVisitorRefinements();
