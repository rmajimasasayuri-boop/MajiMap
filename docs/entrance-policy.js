// Official January schedule, with the October 1 opening-door change. Japan time.
// https://www.med.kindai.ac.jp/notice/assets/2026/01/notice260113.pdf
// https://www.med.kindai.ac.jp/notice/2026_0917_6883.html
const regularEntranceRefs=['hospital:3::southWestEntrance','hospital:3::southEastEntrance','hospital:3::receptionEntrance'];
function entranceAvailability(reference,date=new Date()){
 if(!regularEntranceRefs.includes(reference))return {closed:false,state:'open',label:reference==='hospital:3::afterHoursEntrance'?'24時間利用可能':''};
 const v=Object.fromEntries(new Intl.DateTimeFormat('en-GB',{timeZone:'Asia/Tokyo',year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hourCycle:'h23'}).formatToParts(date).map(p=>[p.type,p.value]));
 const minute=Number(v.hour)*60+Number(v.minute),day=new Date(Date.UTC(+v.year,+v.month-1,+v.day)).getUTCDay(),key=v.month+'-'+v.day;
 const holidays={2026:['01-01','01-12','02-11','02-23','03-20','04-29','05-03','05-04','05-05','05-06','07-20','08-11','09-21','09-22','09-23','10-12','11-03','11-23'],2027:['01-01','01-11','02-11','02-23','03-21','03-22','04-29','05-03','05-04','05-05','07-19','08-11','09-20','09-23','10-11','11-03','11-23']};
 const changed=v.year+'-'+v.month+'-'+v.day>='2026-10-01',early=reference===(changed?'hospital:3::receptionEntrance':'hospital:3::southEastEntrance'),open=early?470:480,close=day===6?780:1080,fmt=n=>Math.floor(n/60)+':'+String(n%60).padStart(2,'0');
 if(day===0||holidays[v.year]?.includes(key))return {closed:true,state:'closed',label:'閉鎖中（日曜・祝日） · 時間外出入口へ'};
 if(!holidays[v.year])return {closed:true,state:'unknown',label:'開門日を要確認 · 時間外出入口へ'};
 const hours=fmt(open)+'–'+fmt(close);
 return minute<open||minute>=close?{closed:true,state:'closed',label:'閉鎖中 · '+(minute<open?'本日'+fmt(open)+'開門':'時間外出入口へ'),hours}:minute>=close-10?{closed:false,state:'soon-close',label:'まもなく閉鎖 · '+fmt(close)+'まで',hours}:{closed:false,state:'open',label:'利用可能 · '+hours,hours};
}
function entranceEdgeAllowed(u,v,date=new Date()){
 // Split corridor graphs can have aliases at the exact same physical doorway.
 const physicalDoor=r=>{const n=(typeof mobilityData!=='undefined'&&mobilityData.nodes[r])||(typeof confirmedNavigation!=='undefined'&&confirmedNavigation.nodes[r]);if(n?.floor!==3)return r;return regularEntranceRefs.find(id=>{const p=mobilityData.points[id];return p&&Math.hypot(p[0]-n.point[0],p[1]-n.point[1])<.1;})||r;};
 return !entranceAvailability(physicalDoor(u),date).closed&&!entranceAvailability(physicalDoor(v),date).closed;
}
function placeNearestKind(d){return d.type==='entrance'?'entrance':d.type==='toilet'?'toilet':d.type==='equipment'?d.category:null;}
