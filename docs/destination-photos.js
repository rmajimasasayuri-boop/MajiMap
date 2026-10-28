const officialPhotoCatalog={"outpatient:1::1A": {"files": ["f1A.webp"], "label": "1A"}, "outpatient:1::1E": {"files": ["f1E.webp"], "label": "1E"}, "outpatient:2::2A": {"files": ["f2A.webp"], "label": "2A"}, "outpatient:2::2B": {"files": ["f2B.webp"], "label": "2B"}, "outpatient:2::2E": {"files": ["f2E.webp"], "label": "2E"}, "outpatient:2::2G": {"files": ["f2G.webp"], "label": "2G"}, "outpatient:2::2K": {"files": ["f2K.webp"], "label": "2K"}, "outpatient:2::2J": {"files": ["f2J.webp"], "label": "2J"}, "outpatient:2::2F": {"files": ["f2F.webp"], "label": "2F"}, "outpatient:2::2S": {"files": ["f2S.webp"], "label": "2S"}, "outpatient:2::2L": {"files": ["f2L.webp"], "label": "2L"}, "outpatient:2::2P": {"files": ["f2P.webp"], "label": "2P"}, "outpatient:2::2R": {"files": ["f2R.webp"], "label": "2R"}, "outpatient:3::3A": {"files": ["f3A.webp"], "label": "3A"}, "outpatient:3::3B": {"files": ["f3B.webp"], "label": "3B"}, "outpatient:3::3C": {"files": ["f3C.webp"], "label": "3C"}, "outpatient:3::3G": {"files": ["f3G.webp"], "label": "3G"}, "outpatient:3::3E": {"files": ["f3E.webp"], "label": "3E"}, "outpatient:3::3P": {"files": ["f3P.webp"], "label": "3P"}, "outpatient:3::3L": {"files": ["f3L.webp"], "label": "3L"}, "outpatient:4::4A": {"files": ["f4A.webp"], "label": "4A"}, "outpatient:4::4E": {"files": ["f4E.webp"], "label": "4E"}, "outpatient:4::4F": {"files": ["f4F.webp"], "label": "4F"}, "outpatient:4::4G": {"files": ["f4G.webp"], "label": "4G"}, "outpatient:4::4L": {"files": ["f4L.webp"], "label": "4L"}, "outpatient:4::4P": {"files": ["f4P.webp"], "label": "4P"}, "outpatient:5::5B": {"files": ["f5B.webp"], "label": "5B"}, "outpatient:5::5C": {"files": ["f5C.webp"], "label": "5C"}, "outpatient:5::5E": {"files": ["f5E.webp"], "label": "5E"}, "outpatient:5::5G": {"files": ["f5G.webp"], "label": "5G"}, "outpatient:5::5L": {"files": ["f5L.webp"], "label": "5L"}, "outpatient:6::6A": {"files": ["f6A.webp"], "label": "6A"}, "outpatient:6::6B": {"files": ["f6B.webp"], "label": "6B"}, "outpatient:6::6G": {"files": ["f6G.webp"], "label": "6G"}, "hospital:3::generalReception": {"files": ["f3info01.webp"]}, "services::seven": {"files": ["f2info01.webp"]}, "services::green": {"files": ["f2info02.webp"]}, "services::information": {"files": ["f3info04.webp"]}, "services::laundry": {"files": ["f3info03.webp"]}, "services::admission": {"files": ["f3info05.webp"]}, "hospital:7::rehab": {"files": ["f7P.webp"]}, "services::cosmetics": {"files": ["pic_cosmetics_store04.webp"]}, "services::glasses": {"files": ["pic_megane_store03.webp"]}, "services::hair": {"files": ["pic_cosmetics_store01.webp"]}, "services::restaurant": {"files": ["pic_cafe0201.webp"]}, "services::cafe": {"files": ["pic_cafe0101.webp"]}};
const officialPhotoBase='https://www.med.kindai.ac.jp/img/floormap_facility/floor/';
const officialPhotoSource='https://www.med.kindai.ac.jp/floormap_facility.html';
function officialPhotosFor(d){const entry=d&&officialPhotoCatalog[d.ref];return entry?entry.files.map(file=>({url:officialPhotoBase+file,source:officialPhotoSource})):[];}
function appendPlaceResultPhoto(row,d){
 const photo=officialPhotosFor(d)[0];if(!photo)return;
 const image=document.createElement('img');image.className='place-result-photo';image.alt='';image.title='近畿大学病院 公式写真';image.loading='lazy';image.decoding='async';image.referrerPolicy='no-referrer';image.width=64;image.height=48;image.onerror=()=>image.remove();image.src=photo.url;row.append(image);
}
let destinationPhotoState={place:null,photos:[],index:0};
function updateDestinationPhoto(d){if(typeof hidePlacePreview==='function')hidePlacePreview();renderPlacePhoto(d);}
function renderPlacePhoto(d){
 const photos=officialPhotosFor(d),preview=document.getElementById('destinationPhotoPreview'),image=document.getElementById('destinationPhotoThumb');
 destinationPhotoState={place:d,photos,index:0};if(typeof updateRoutePlanner==='function')updateRoutePlanner();preview.hidden=true;const panel=document.getElementById('facilityDetails');panel.hidden=!d;if(!d)return;document.getElementById('facilityDetailName').textContent=destinationDisplayName(d);document.getElementById('facilityDetailFloor').textContent=d.floor+'階'+(d.code?' · '+d.code:'');document.getElementById('detailWebsite').href=d.detail?.sourceUrl||officialPhotoSource;document.getElementById('detailRoute').disabled=isGuideOnly(d);document.getElementById('detailStart').disabled=isGuideOnly(d);document.getElementById('detailNavigate').disabled=isGuideOnly(d);document.getElementById('detailNavigate').textContent=isGuideOnly(d)?'通路未登録':'ナビを開始';
 if(!photos.length){image.removeAttribute('src');return;}
 const name=destinationDisplayName(d);document.getElementById('destinationPhotoLabel').textContent=name;
 image.alt=name+'の公式写真';document.getElementById('viewDestinationPhoto').setAttribute('aria-label',name+'の写真を見る');
 image.onload=()=>{if(destinationPhotoState.place?.ref===d.ref)preview.hidden=false;};image.onerror=()=>{preview.hidden=true;};image.src=photos[0].url;
}
function showDestinationPhoto(){
 const {place,photos,index}=destinationPhotoState;if(!photos.length)return;
 document.getElementById('destinationPhotoTitle').textContent=destinationDisplayName(place);
 const image=document.getElementById('destinationPhotoFull');image.alt=destinationDisplayName(place)+'（近畿大学病院公式フロアマップ掲載写真）';
 document.getElementById('destinationPhotoError').hidden=true;image.onerror=()=>{document.getElementById('destinationPhotoError').hidden=false;};image.src=photos[index].url;
 document.getElementById('destinationPhotoSource').href=photos[index].source;
 const dialog=document.getElementById('destinationPhotoDialog');if(!dialog.open)dialog.showModal();
}
document.getElementById('viewDestinationPhoto').onclick=showDestinationPhoto;
document.getElementById('closeDestinationPhoto').onclick=()=>document.getElementById('destinationPhotoDialog').close();

const facilityDetails=document.getElementById('facilityDetails');
document.getElementById('closeFacilityDetails').onclick=()=>{facilityDetails.hidden=true;hidePlacePreview();};
document.getElementById('detailStart').onclick=()=>{const d=destinationPhotoState.place;if(d){selectStartPlace(d,true);updateDestinationPhoto(null);}};
document.getElementById('detailRoute').onclick=()=>{const d=destinationPhotoState.place;if(!d)return;choosePlace(d,true);if(!ensureIndoorStart())return;routePlan();facilityDetails.hidden=true;};
document.getElementById('detailNavigate').onclick=()=>{const d=destinationPhotoState.place;if(d){choosePlace(d,true);if(!ensureIndoorStart())return;routePlan();if(navigationState.steps.length)startNavigation();}};
let facilityTouch=null;
facilityDetails.addEventListener('pointerdown',e=>{e.stopPropagation();if(e.target.closest('button'))return;facilityDetails.setPointerCapture(e.pointerId);facilityTouch={x:e.clientX,y:e.clientY};});
facilityDetails.addEventListener('pointerup',e=>{if(!facilityTouch)return;const dy=e.clientY-facilityTouch.y,dx=e.clientX-facilityTouch.x;facilityTouch=null;if(Math.abs(dy)>45&&Math.abs(dy)>Math.abs(dx)){if(dy>0)facilityDetails.classList.remove('expanded');else facilityDetails.classList.add('expanded');}});
facilityDetails.addEventListener('pointercancel',()=>{facilityTouch=null;});
