// Prefer a position near the facility, excluding the route/settings controls.
function placePreviewPlacement(anchor,size,bounds,obstacles){
 const [w,h]=size,pad=8,maxX=bounds[0]-w-pad,maxY=bounds[1]-h-pad;
 if(maxX<pad||maxY<pad)return null;
 const xs=[anchor[0]+14,anchor[0]-w-14,anchor[0]-w/2,pad,maxX],ys=[anchor[1]+14,anchor[1]-h-14,anchor[1]-h/2,pad,maxY];
 for(const r of obstacles){xs.push(r.x-w-8,r.x+r.width+8);ys.push(r.y-h-8,r.y+r.height+8);}
 let best=null;
 for(const x0 of xs)for(const y0 of ys){const x=Math.max(pad,Math.min(maxX,x0)),y=Math.max(pad,Math.min(maxY,y0));
  if(obstacles.some(r=>x<r.x+r.width+4&&x+w>r.x-4&&y<r.y+r.height+4&&y+h>r.y-4))continue;
  const dx=Math.max(x-anchor[0],0,anchor[0]-x-w),dy=Math.max(y-anchor[1],0,anchor[1]-y-h),score=dx*dx+dy*dy;
  if(!best||score<best.score)best={x,y,score};
 }
 return best;
}
let mapPlacePreview={place:null,anchor:null,pinned:false,mode:'goal'},photoRestoreTimer=0;
function hidePlacePreview(){clearTimeout(photoRestoreTimer);mapPlacePreview.place=null;document.getElementById('map3DHover').hidden=true;}
function schedulePhotoRestore(){clearTimeout(photoRestoreTimer);photoRestoreTimer=setTimeout(restoreDestinationPhoto,180);}
function restoreDestinationPhoto(){if(!mapPlacePreview.pinned)hidePlacePreview();}
function positionPlacePreview(){
 const card=document.getElementById('map3DHover');if(card.hidden||!mapPlacePreview.anchor)return;
 const viewport=document.getElementById('mapViewport').getBoundingClientRect(),a=mapPlacePreview.anchor;
 const anchor=[a.x-viewport.left,a.y-viewport.top],obstacles=[];
 for(const selector of ['#facilityDetails','#settings','#navigationPanel','#mapSearchBar','#nearbyEquipmentBar','.map-heading','.floor-toolbar','#map3DControls']){
  const e=document.querySelector(selector);if(!e||e.hidden||!e.getClientRects().length||getComputedStyle(e).display==='none'||getComputedStyle(e).visibility==='hidden')continue;
  const r=e.getBoundingClientRect();obstacles.push({x:r.left-viewport.left,y:r.top-viewport.top,width:r.width,height:r.height});
 }
 card.classList.remove('preview-compact');card.style.width=Math.min(280,viewport.width-16)+'px';
 let point=placePreviewPlacement(anchor,[card.offsetWidth,card.offsetHeight],[viewport.width,viewport.height],obstacles);
 if(!point){card.classList.add('preview-compact');card.style.width=Math.min(220,viewport.width-16)+'px';point=placePreviewPlacement(anchor,[card.offsetWidth,card.offsetHeight],[viewport.width,viewport.height],obstacles);}
 if(!point){card.hidden=true;return;}card.style.left=point.x+'px';card.style.top=point.y+'px';
}
function previewPlacePhoto(d,anchor,pinned=false){
 clearTimeout(photoRestoreTimer);if(!d||mapPlacePreview.pinned&&!pinned)return;
 const point=anchor?{x:anchor.x+(anchor.width||0)/2,y:anchor.y+(anchor.height||0)/2}:null;
 if(!point)return;
 const changed=d.ref!==mapPlacePreview.place?.ref,card=document.getElementById('map3DHover');
 mapPlacePreview={place:d,anchor:point,pinned,mode:mapUI.pickMode};
 document.getElementById('map3DHoverName').textContent=d.floor+'階 · '+destinationDisplayName(d);
 document.getElementById('map3DHoverAction').textContent='クリックで'+(mapUI.pickMode==='start'?'出発地':'目的地')+'に設定';card.dataset.mode=mapUI.pickMode;
 const image=document.getElementById('placePreviewPhoto'),source=document.getElementById('placePreviewSource'),photo=officialPhotosFor(d)[0];
 if(changed){image.hidden=true;source.hidden=true;if(photo){image.alt=destinationDisplayName(d)+'の公式写真';image.onload=()=>{if(mapPlacePreview.place?.ref===d.ref){image.hidden=false;source.hidden=false;positionPlacePreview();}};image.onerror=()=>{image.hidden=true;source.hidden=true;positionPlacePreview();};image.src=photo.url;}else image.removeAttribute('src');}
 card.hidden=false;positionPlacePreview();
}
const placePreviewCard=document.getElementById('map3DHover');
placePreviewCard.onpointerenter=()=>clearTimeout(photoRestoreTimer);placePreviewCard.onpointerleave=schedulePhotoRestore;
placePreviewCard.addEventListener('pointerdown',e=>e.stopPropagation());
document.getElementById('closePlacePreview').onclick=hidePlacePreview;
document.getElementById('map3DHoverAction').onclick=()=>{const d=mapPlacePreview.place;hidePlacePreview();if(d)selectPlace(d,true);};
window.addEventListener('resize',hidePlacePreview);
