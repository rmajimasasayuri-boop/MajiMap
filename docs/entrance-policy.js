// User-confirmed hospital entrance schedule. Japan time; not clinical hours.
const regularEntranceRefs=['hospital:3::southWestEntrance','hospital:3::southEastEntrance','hospital:3::receptionEntrance'];
function entranceAvailability(reference,date=new Date()){
 if(!regularEntranceRefs.includes(reference))return {closed:false,state:'open',label:reference==='hospital:3::afterHoursEntrance'?'24時間利用可能':''};
 const parts=new Intl.DateTimeFormat('en-GB',{timeZone:'Asia/Tokyo',hour:'2-digit',minute:'2-digit',hourCycle:'h23'}).formatToParts(date),v=Object.fromEntries(parts.map(p=>[p.type,p.value])),minute=Number(v.hour)*60+Number(v.minute);
 return minute<480||minute>=1080?{closed:true,state:'closed',label:'閉鎖中 · 8:00に開放'}:minute>=1070?{closed:false,state:'soon-close',label:'まもなく閉鎖 · 18:00まで'}:{closed:false,state:'open',label:'利用可能 · 8:00–18:00'};
}
function entranceEdgeAllowed(u,v,date=new Date()){return !entranceAvailability(u,date).closed&&!entranceAvailability(v,date).closed;}
function placeNearestKind(d){return d.type==='entrance'?'entrance':d.type==='toilet'?'toilet':d.type==='equipment'?d.category:null;}
