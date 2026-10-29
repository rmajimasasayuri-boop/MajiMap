// Visitor-facing readability, preview cards and one editable waypoint.
function closeVisitorPreview(){mapUI.previewPlace=null;document.body.classList.remove('place-preview-open');ui('facilityDetails').hidden=true;if(typeof renderOverlay==='function')renderOverlay();}
function showVisitorPreview(d){
 mapUI.previewPlace=d;renderPlacePhoto(d);document.body.classList.add('place-preview-open');ui('facilityDetails').hidden=false;updatePreviewActions();renderOverlay();
}
function updatePreviewActions(){
 const d=destinationPhotoState.place,b=ui('detailNavigate');b.innerHTML='<span>経由地を<br>追加</span>';b.disabled=!d||!mapUI.goal||d.ref===mapUI.goal.ref||isGuideOnly(d)||d.type==='manual';b.title=!mapUI.goal?'先に目的地を設定してください':d?.ref===mapUI.goal.ref?'この場所は目的地です':'';
}
function addVisitorWaypoint(){
 const d=destinationPhotoState.place;if(!d||!mapUI.goal||isGuideOnly(d)||d.ref===mapUI.goal.ref)return;
 stopNavigation();mapUI.waypoint=d;rememberPlace(d);closeVisitorPreview();enterRoutePlanning();syncVisitorRefinements();
}
function syncVisitorRefinements(){
 updatePreviewActions();
 const row=ui('routeViaRow');if(row){row.hidden=!mapUI.waypoint;if(mapUI.waypoint)ui('routeViaLabel').textContent='経由：'+destinationDisplayName(mapUI.waypoint);}
 ui('chooseCurrentMapPoint').hidden=true;
 requestAnimationFrame(()=>{const box=ui('routePlanner').getBoundingClientRect();if(box.height)document.documentElement.style.setProperty('--route-panel-bottom',(box.bottom+12)+'px');});
}
function boundedVisitorPan(pan,size,scale,view,area,bounds={x:0,y:0,w:size[0],h:size[1]}){
 const origin=[view.width/2-size[0]*scale/2+pan[0]+bounds.x*scale,view.height/2-size[1]*scale/2+pan[1]+bounds.y*scale],extent=[bounds.w*scale,bounds.h*scale];
 return pan.map((v,i)=>{const first=i?area.top:area.left,last=i?area.bottom:area.right,length=last-first;const next=extent[i]<=length?first+(length-extent[i])/2:Math.max(last-extent[i]-20,Math.min(first+20,origin[i]));return v+next-origin[i];});
}
function constrainVisitorPan(){
 if(document.body.classList.contains('view-3d'))return mapUI.pan;
 const v=ui('mapViewport'),box=v.getBoundingClientRect(),width=v.clientWidth,height=v.clientHeight;let top=66,bottom=height-44;
 for(const id of ['routePlanner','facilityDetails','navigationPanel']){const e=ui(id),r=e.getBoundingClientRect();if(!r.height||getComputedStyle(e).display==='none')continue;if(id==='routePlanner')top=Math.max(top,r.bottom-box.top+10);else bottom=Math.min(bottom,r.top-box.top-10);}
 top=Math.min(top,height*.45);bottom=Math.max(bottom,top+120);const area={left:12,right:Math.max(160,width-76),top,bottom:Math.min(height-36,bottom)};
 return boundedVisitorPan(mapUI.pan,floorInfo(mapUI.floor).size,mapUI.base*mapUI.zoom,{width,height},area,mapUI.floor>=7?{x:897,y:183,w:823,h:829}:undefined);
}
const visitorFavoritesKey='maji-place-favorites-v1';
function readPlaceFavorites(){try{const list=JSON.parse(localStorage.getItem(visitorFavoritesKey)||'[]');return Array.isArray(list)?[...new Set(list.filter(ref=>typeof ref==='string'&&destinationList.some(d=>d.ref===ref)))]:[];}catch{return [];}}
function updateVisitorFavorites(){
 const list=ui('favoritePlaces');if(!list)return;const refs=readPlaceFavorites();list.replaceChildren();for(const ref of refs){const d=destinationList.find(p=>p.ref===ref),row=document.createElement('div');row.className='favorite-place-row';const select=document.createElement('button');select.textContent=d.floor+'階 '+destinationDisplayName(d);select.onclick=()=>selectPlace(d);const remove=document.createElement('button');remove.textContent='☆';remove.setAttribute('aria-label',destinationDisplayName(d)+'のお気に入りを解除');remove.onclick=()=>toggleVisitorFavorite(d);row.append(select,remove);list.append(row);}
 ui('favoriteEmpty').hidden=Boolean(refs.length);const d=destinationPhotoState.place,b=ui('favoritePlaceButton'),saved=d&&refs.includes(d.ref);b.textContent=saved?'★ お気に入り登録済み':'☆ お気に入りに登録';b.setAttribute('aria-pressed',String(Boolean(saved)));b.disabled=!d||d.type==='manual';b.title='このブラウザ・サイトに保存します';
}
function toggleVisitorFavorite(d=destinationPhotoState.place){if(!d||d.type==='manual')return;const refs=readPlaceFavorites(),next=refs.includes(d.ref)?refs.filter(ref=>ref!==d.ref):[d.ref,...refs];try{localStorage.setItem(visitorFavoritesKey,JSON.stringify(next));ui('favoriteStatus').textContent=next.includes(d.ref)?'お気に入りに登録しました。':'お気に入りを解除しました。';updateVisitorFavorites();}catch{ui('favoritePlaceButton').textContent='保存できませんでした';ui('favoriteStatus').textContent='このブラウザでは保存できません。';}}
function initVisitorFavorites(){
 const section=document.createElement('section');section.className='favorite-section';section.innerHTML='<h3>お気に入り</h3><p id="favoriteStatus" role="status">よく行く外来や病棟を保存し、出発地・目的地に選べます。</p><div id="favoritePlaces"></div><p id="favoriteEmpty">施設の写真カードの☆から登録できます。</p>';ui('searchChoices').append(section);
 const b=document.createElement('button');b.id='favoritePlaceButton';b.className='favorite-place-button';b.onclick=()=>toggleVisitorFavorite();ui('facilityDetails').querySelector('.facility-detail-body').append(b);updateVisitorFavorites();
}
function initVisitorRefinements(){
 if(reviewEmbedded)return;
 const originalPhoto=renderPlacePhoto;renderPlacePhoto=function(d){originalPhoto(d);updatePreviewActions();updateVisitorFavorites();};
 const originalHide=hidePlacePreview;hidePlacePreview=function(){originalHide();mapUI.previewPlace=null;document.body.classList.remove('place-preview-open');};
 previewPlacePhoto=function(d,anchor,pinned=false){if(!pinned||!d)return;if(mapUI.pickMode==='start'){selectStartPlace(d,true);return;}showVisitorPreview(d);};
 ui('detailRoute').onclick=()=>{const d=destinationPhotoState.place;if(!d)return;closeVisitorPreview();choosePlace(d,true);rememberPlace(d);enterRoutePlanning();};
 ui('detailNavigate').onclick=addVisitorWaypoint;ui('detailWebsite').innerHTML='<span>ウェブ<br>サイト</span>';ui('detailWebsite').setAttribute('aria-label','ウェブサイト');
 ui('closeFacilityDetails').onclick=closeVisitorPreview;
 const row=document.createElement('div');row.id='routeViaRow';row.className='route-via-row';row.hidden=true;const label=document.createElement('strong');label.id='routeViaLabel';const remove=document.createElement('button');remove.textContent='×';remove.setAttribute('aria-label','経由地を削除');remove.onclick=()=>{mapUI.waypoint=null;routePlan();syncVisitorRefinements();};row.append(label,remove);ui('routePlanner').querySelector('.route-place-fields').append(row);
 const home=ui('map3DHome');home.classList.add('floor-home-button');document.querySelector('.map-navigation-rail').append(home);
 const wheelchair=document.querySelector('[data-travel-choice="elevator"]');wheelchair.setAttribute('aria-label','車椅子（EVのみ）');wheelchair.title='階の移動にはエレベーターのみを使います';for(const node of [...wheelchair.childNodes])if(node.nodeType===3)node.textContent='車椅子';
 // Equipment icons are subfilters shown only inside Equipment, avoiding duplicate categories.
 const equipment=ui('nearbyEquipmentBar');equipment.classList.add('equipment-subfilters');equipment.setAttribute('aria-label','設備の種類');
 const originalResults=renderResults;renderResults=function(){originalResults();equipment.hidden=mapUI.filter!=='equipment';};
 const originalCategory=resultCategoryMatches;resultCategoryMatches=function(d){if(mapUI.filter==='amenity')return ['services::restaurant','services::cafe','services::seven','services::green','services::cosmetics','services::glasses','services::hair'].includes(d.ref);return originalCategory(d);};
 window.addEventListener('maji-map-update',syncVisitorRefinements);new ResizeObserver(syncVisitorRefinements).observe(ui('routePlanner'));
 initVisitorFavorites();updatePreviewActions();renderResults();syncVisitorRefinements();applyMapTransform();
}
initVisitorRefinements();
