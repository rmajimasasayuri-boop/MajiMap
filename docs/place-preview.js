// Facilities highlight in place; selecting opens the bottom sheet.
function hidePlacePreview(){document.getElementById('map3DHover').hidden=true;}
function schedulePhotoRestore(){}
function previewPlacePhoto(d,anchor,pinned=false){if(!pinned||!d)return;hidePlacePreview();if(mapUI.pickMode==='start')selectStartPlace(d,true);else{choosePlace(d,true);updateRoutePlanner();} }
