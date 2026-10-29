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
function initVisitorRefinements(){
 if(reviewEmbedded)return;
 const originalPhoto=renderPlacePhoto;renderPlacePhoto=function(d){originalPhoto(d);updatePreviewActions();};
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
 updatePreviewActions();renderResults();syncVisitorRefinements();
}
initVisitorRefinements();
