"use strict";
/* ================= КОНСТАНТЫ ПРОЕКТА ================= */
const RAW="/dashboard/data/";
const CLASSES=["Bumps","Dirtiness","K_Scatch","Other_Faults","Pastry","Stains","Z_Scratch"];
const RU={Bumps:"Выпуклости",Dirtiness:"Загрязнение",K_Scatch:"K-царапина",Other_Faults:"Прочее",Pastry:"Наплыв",Stains:"Пятна",Z_Scratch:"Z-царапина"};
const COLOR={Bumps:"#b8860b",Dirtiness:"#5c8a3c",K_Scatch:"#c0392b",Other_Faults:"#8c7b70",Pastry:"#ff8e45",Stains:"#7b5ea7",Z_Scratch:"#2f7d8c"};
const BRIGHTNESS=["Minimum_of_Luminosity","Maximum_of_Luminosity","Sum_of_Luminosity","Luminosity_Index"];
const MIN_BATCH=30, SIZES=[30,60,100,150,200], DRAWS=96, QLEVEL=.95;
const SEEDS=[42,47,53], BUDGETS=[0,10,20,40,80];
const MODEL_NAMES={xgb:"XGBoost",hist:"Gradient boosting (hist)",smote:"XGBoost + SMOTE",rf:"Random Forest"};
const STATUS={shift:"Есть сдвиг",stable:"Без сдвига",insufficient_data:"Малая партия"};

/* ================= УТИЛИТЫ ================= */
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const clamp=(v,a,b)=>Math.min(b,Math.max(a,v));
const f3=v=>Number(v).toFixed(3), f2=v=>Number(v).toFixed(2), pc=v=>(v*100).toFixed(1)+" %";
const esc=s=>String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
function quantileSorted(sorted,p,method){const n=sorted.length;if(!n)return NaN;
  if(method==="higher")return sorted[clamp(Math.ceil(p*n)-1,0,n-1)];
  const h=(n-1)*p,lo=Math.floor(h),hi=Math.ceil(h);return sorted[lo]+(sorted[hi]-sorted[lo])*(h-lo)}
function toast(msg,kind){const el=document.createElement("div");el.className="toast";
  el.style.borderLeftColor=kind==="err"?"#c0392b":kind==="ok"?"#3f8f63":"var(--orange)";
  el.innerHTML="<div style='flex:1'>"+msg+"</div>";$("#toasts").appendChild(el);
  setTimeout(()=>{el.classList.add("out");setTimeout(()=>el.remove(),240)},4200)}
async function copyText(t){try{await navigator.clipboard.writeText(t);return true}catch(e){
  const ta=document.createElement("textarea");ta.value=t;ta.style.position="fixed";ta.style.opacity="0";
  document.body.appendChild(ta);ta.select();const ok=document.execCommand("copy");ta.remove();return ok}}
function download(name,text,type){const blob=new Blob([text],{type:type||"text/plain;charset=utf-8"});
  const u=URL.createObjectURL(blob),a=document.createElement("a");a.href=u;a.download=name;
  document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(u),1200)}
const csvBom=s=>"\ufeff"+s;
function parseCsv(text){
  const rows=[];let row=[],cell="",q=false;
  text=text.replace(/^\ufeff/,"");
  for(let i=0;i<text.length;i++){const c=text[i];
    if(q){if(c==='"'){if(text[i+1]==='"'){cell+='"';i++}else q=false}else cell+=c}
    else if(c==='"')q=true;
    else if(c===","||c===";"){row.push(cell);cell=""}
    else if(c==="\n"){row.push(cell);rows.push(row);row=[];cell=""}
    else if(c!=="\r")cell+=c}
  if(cell.length||row.length){row.push(cell);rows.push(row)}
  return rows.filter(r=>r.length>1||r[0]!=="")}
function csvObjects(text){const r=parseCsv(text);if(!r.length)return[];
  const head=r[0].map(h=>h.trim());
  return r.slice(1).map(cells=>Object.fromEntries(head.map((h,i)=>[h,(cells[i]??"").trim()])))}

/* ================= СОСТОЯНИЕ ================= */
const S={
  names:[],features:[],targets:[],rows:[],labels:[],
  split:null,baseline:null,benchmark:null,holdout:[],
  batches:[],batchId:null,drift:null,qModel:"xgb",queue:[],entry:null,
  labelsStore:{},demo:null,
  sources:{nna:"idle",var:"idle",bench:"idle",hold:"idle"},calibrated:false
};
const LS="steel-qc-dashboard";
function persist(){try{localStorage.setItem(LS,JSON.stringify({labelsStore:S.labelsStore,qModel:S.qModel}))}catch(e){}}
function restore(){try{const s=JSON.parse(localStorage.getItem(LS)||"{}");
  if(s.labelsStore)S.labelsStore=s.labelsStore;if(s.qModel)S.qModel=s.qModel}catch(e){}}

/* ================= ЗАГРУЗКА ДАННЫХ ================= */
async function fetchText(path){const r=await fetch(RAW+path,{cache:"no-store"});
  if(!r.ok)throw new Error("HTTP "+r.status);return r.text()}
function setSrc(key,state,note){S.sources[key]=state;renderSources(note)}
function renderSources(){
  const defs=[["var","Faults27x7_var","имена 27 признаков и 7 категорий"],
    ["nna","Faults.NNA","1941 запись · 34 столбца"],
    ["bench","analysis/benchmark.json","сравнение 6 моделей × 3 семени"],
    ["hold","analysis/holdout_predictions.csv","389 прогнозов четырёх моделей"]];
  $("#srcList").innerHTML=defs.map(([k,name,desc])=>{
    const st=S.sources[k];const cls=st==="ok"?"ok":st==="bad"?"bad":st==="load"?"load":"";
    const label=st==="ok"?"загружен":st==="bad"?"недоступен":st==="load"?"загрузка…":"ожидание";
    return '<div class="src-item '+cls+'"><span class="dot"></span><div style="flex:1;min-width:0"><b>'+name+'</b><small>'+desc+'</small></div><span class="badge grey">'+label+'</span></div>'}).join("")}

function parseNNA(text,names){
  const lines=text.split(/\r?\n/).filter(l=>l.trim());
  const rows=lines.map(l=>l.split(/\t|,|;|\s+/).filter(x=>x!=="").map(Number));
  const width=names.length;
  return rows.filter(r=>r.length>=width).map(r=>r.slice(0,width))}

function buildDataset(){
  const nf=S.names.length-7;
  S.features=S.names.slice(0,nf);S.targets=S.names.slice(nf);
  S.rows=S.matrix.map(r=>Object.fromEntries(S.features.map((f,i)=>[f,r[i]])));
  S.labels=S.matrix.map(r=>{const t=r.slice(nf);const i=t.indexOf(Math.max(...t));return S.targets[i]});
  S.split=stratifiedSplit(S.labels,42)}

function stratifiedSplit(labels,seed){
  const rng=mulberry32(seed),byClass={};
  labels.forEach((l,i)=>{(byClass[l]=byClass[l]||[]).push(i)});
  const train=[],validation=[],pool=[],test=[];
  Object.keys(byClass).sort().forEach(c=>{
    const idx=byClass[c].slice();
    for(let i=idx.length-1;i>0;i--){const j=Math.floor(rng()*(i+1));[idx[i],idx[j]]=[idx[j],idx[i]]}
    const n=idx.length,nTest=Math.round(n*.20),nTrain=Math.round((n-nTest)*.5);
    const rest=n-nTest-nTrain,nVal=Math.round(rest*.5);
    test.push(...idx.slice(0,nTest));train.push(...idx.slice(nTest,nTest+nTrain));
    validation.push(...idx.slice(nTest+nTrain,nTest+nTrain+nVal));
    pool.push(...idx.slice(nTest+nTrain+nVal))});
  const srt=a=>a.sort((x,y)=>x-y);
  return{train:srt(train),validation:srt(validation),pool:srt(pool),test:srt(test)}}

/* ================= KS И ПОРОГИ (faults/drift.py) ================= */
function ksSorted(ref,sam){
  const n=ref.length,m=sam.length;let i=0,j=0,max=0;
  while(i<n||j<m){
    const v=(j>=m||(i<n&&ref[i]<=sam[j]))?ref[i]:sam[j];
    while(i<n&&ref[i]<=v)i++;
    while(j<m&&sam[j]<=v)j++;
    const d=Math.abs(i/n-j/m);if(d>max)max=d}
  return max}
function columnSorted(indices,col){
  const a=Float64Array.from(indices,i=>S.rows[i][col]);a.sort();return a}
function sampleMatrix(indices){
  return S.features.map(f=>{const a=Float64Array.from(indices,i=>S.rows[i][f]);a.sort();return a})}

async function calibrate(){
  if(!S.split)return;
  const bar=$("#calibBar");bar.hidden=false;
  const ref=S.features.map(f=>columnSorted(S.split.train,f));
  const val=S.split.validation, thresholds={}, rng=mulberry32(42);
  const usable=SIZES.filter(s=>s<=val.length);
  let done=0;
  for(const size of usable){
    const maxima=[];
    for(let d=0;d<DRAWS;d++){
      const pick=[];for(let k=0;k<size;k++)pick.push(val[Math.floor(rng()*val.length)]);
      const sam=sampleMatrix(pick);
      let mx=0;for(let c=0;c<ref.length;c++){const v=ksSorted(ref[c],sam[c]);if(v>mx)mx=v}
      maxima.push(mx);
      if(d%12===0){bar.firstElementChild.style.width=(++done/(usable.length*8)*100)+"%";
        await new Promise(r=>setTimeout(r,0))}}
    maxima.sort((a,b)=>a-b);
    thresholds[size]=quantileSorted(maxima,QLEVEL,"higher");
    bar.firstElementChild.style.width=(done/(usable.length*8)*100)+"%";
    await new Promise(r=>setTimeout(r,0))}
  S.baseline={columns:S.features.slice(),reference:ref,thresholds};
  S.calibrated=true;bar.hidden=true;
  $("#calibNote").textContent="Пороги откалиброваны в браузере на "+S.split.train.length+" опорных и "+val.length+" калибровочных записях. Разбиение стратифицированное, seed 42 — воспроизведение split_dataset; серверные пороги могут отличаться в третьем знаке.";
  assessCurrent()}

function assessDrift(batchRows){
  if(!S.baseline)return{status:"insufficient_data",sample_size:batchRows.length,scores:[]};
  if(batchRows.length<MIN_BATCH)return{status:"insufficient_data",sample_size:batchRows.length,scores:[]};
  const sizes=Object.keys(S.baseline.thresholds).map(Number);
  const size=Math.max(...sizes.filter(s=>s<=batchRows.length));
  let idx=batchRows.map((_,i)=>i);
  if(batchRows.length>size){const rng=mulberry32(42);const pick=new Set();
    while(pick.size<size)pick.add(Math.floor(rng()*batchRows.length));idx=[...pick]}
  const scores=S.features.map((f,c)=>{
    const a=Float64Array.from(idx,i=>batchRows[i].features[f]);a.sort();
    return{feature:f,score:ksSorted(S.baseline.reference[c],a)}});
  scores.sort((a,b)=>b.score-a.score);
  const score=scores[0].score,threshold=S.baseline.thresholds[size];
  return{status:score>threshold?"shift":"stable",score,threshold,sample_size:size,scores,total:batchRows.length}}

function shiftBrightness(batchRows){
  return batchRows.map(r=>{const f=Object.assign({},r.features);
    BRIGHTNESS.forEach(name=>{
      const c=S.features.indexOf(name);if(c<0)return;
      const ref=S.baseline.reference[c];
      const q1=quantileSorted(ref,.25),q3=quantileSorted(ref,.75);
      let mean=0;for(let i=0;i<ref.length;i++)mean+=ref[i];mean/=ref.length;
      let varr=0;for(let i=0;i<ref.length;i++)varr+=(ref[i]-mean)**2;
      const sd=Math.sqrt(varr/ref.length);
      f[name]=f[name]+1.4*Math.max(q3-q1,sd/2)});
    return Object.assign({},r,{features:f})})}

async function runDriftDemo(){
  if(!S.baseline){toast("Дождитесь калибровки порогов","err");return}
  const pool=S.split.pool.map(i=>({features:S.rows[i]}));
  const sizes=Object.keys(S.baseline.thresholds).map(Number).filter(s=>s<=Math.min(pool.length,60));
  if(!sizes.length){toast("Для прогона нужно минимум 30 записей в пуле","err");return}
  const size=Math.max(...sizes),rng=mulberry32(42),trials=40;
  const bar=$("#demoBar");bar.hidden=false;$("#runDemo").disabled=true;
  let normal=0,shifted=0;
  for(let t=0;t<trials;t++){
    const pick=[];for(let k=0;k<size;k++)pick.push(pool[Math.floor(rng()*pool.length)]);
    if(assessDrift(pick).status==="shift")normal++;
    if(assessDrift(shiftBrightness(pick)).status==="shift")shifted++;
    bar.firstElementChild.style.width=((t+1)/trials*100)+"%";
    if(t%4===0)await new Promise(r=>setTimeout(r,0))}
  bar.hidden=true;$("#runDemo").disabled=false;
  S.demo={detection_rate:shifted/trials,false_alert_rate:normal/trials,trials,batch_size:size};
  renderDemo();
  toast("Прогон завершён: обнаружено "+shifted+" из "+trials+" сдвигов","ok")}

/* ================= МЕТРИКИ ================= */
function confusion(truth,pred){
  const idx=Object.fromEntries(CLASSES.map((c,i)=>[c,i]));
  const M=CLASSES.map(()=>CLASSES.map(()=>0));
  truth.forEach((t,i)=>{const a=idx[t],b=idx[pred[i]];if(a!==undefined&&b!==undefined)M[a][b]++});
  const per=CLASSES.map((c,i)=>{
    const tp=M[i][i];let fp=0,fn=0,sup=0;
    for(let j=0;j<7;j++){if(j!==i)fp+=M[j][i];if(j!==i)fn+=M[i][j];sup+=M[i][j]}
    const p=tp+fp?tp/(tp+fp):0,r=tp+fn?tp/(tp+fn):0,f1=p+r?2*p*r/(p+r):0;
    return{cls:c,precision:p,recall:r,f1,support:sup}});
  const correct=truth.filter((t,i)=>t===pred[i]).length;
  return{M,per,accuracy:truth.length?correct/truth.length:0,
    macro_f1:per.reduce((s,x)=>s+x.f1,0)/per.length,size:truth.length}}

/* ================= ОЧЕРЕДЬ (faults/review.py) ================= */
function rankForReview(rows,limit){
  const n=rows.length;limit=clamp(limit,0,n);if(!limit)return[];
  const margins=rows.map(r=>r.margin??0);
  const order=margins.map((m,i)=>[m,i]).sort((a,b)=>a[0]-b[0]||a[1]-b[1]);
  const cand=order.slice(0,Math.max(limit*4,20)).map(x=>x[1]);
  const refIdx=S.split?S.split.train:S.rows.map((_,i)=>i);
  const cols=S.features;
  const med=[],scale=[];
  cols.forEach(f=>{const a=Float64Array.from(refIdx,i=>S.rows[i][f]);a.sort();
    const q1=quantileSorted(a,.25),q3=quantileSorted(a,.75);
    let mean=0;for(let i=0;i<a.length;i++)mean+=a[i];mean/=a.length;
    let v=0;for(let i=0;i<a.length;i++)v+=(a[i]-mean)**2;
    const sd=Math.sqrt(v/a.length);
    med.push(quantileSorted(a,.5));scale.push(q3>q1?q3-q1:Math.max(sd,1))});
  const data=cand.map(i=>cols.map((f,c)=>clamp((rows[i].features[f]-med[c])/scale[c],-8,8)));
  const dist=(p)=>data.map(v=>{let s=0;for(let c=0;c<v.length;c++)s+=(v[c]-data[p][c])**2;
    return Math.sqrt(s)/Math.sqrt(cols.length)});
  const selected=[0];let nearest=dist(0);
  const avail=data.map((_,i)=>i!==0);
  for(let k=1;k<limit;k++){
    let best=-1,bestVal=-Infinity;
    for(let i=0;i<data.length;i++){if(!avail[i])continue;
      const val=1-margins[cand[i]]+.35*Math.min(nearest[i],3);
      if(val>bestVal){bestVal=val;best=i}}
    if(best<0)break;
    avail[best]=false;selected.push(best);
    const d=dist(best);nearest=nearest.map((v,i)=>Math.min(v,d[i]))}
  return selected.map((s,k)=>({position:cand[s],priority:1-margins[cand[s]]+.35*Math.min(k===0?0:nearest[s],3)}))}

/* ================= ПАРТИИ ================= */
function holdoutBatch(model){
  return S.holdout.map(h=>{
    const line=Number(h.source_line)-1;
    const features=S.rows[line]||null;
    if(!features)return null;
    return{position:line,features,pred:h[model],score:null,
      margin:Number(h[model+"_margin"]),truth:h.truth,source_line:Number(h.source_line)}}).filter(Boolean)}

function addBatch(id,label,rows,note){
  S.batches=S.batches.filter(b=>b.id!==id);
  S.batches.push({id,label,rows,note});S.batchId=id;renderBatchSelect();assessCurrent()}
function currentBatch(){return S.batches.find(b=>b.id===S.batchId)||null}
function assessCurrent(){
  const b=currentBatch();
  if(!b){S.drift=null;renderDrift();return}
  S.drift=assessDrift(b.rows);
  buildQueue();renderAll()}

function buildQueue(){
  const b=currentBatch();if(!b){S.queue=[];return}
  const done=S.labelsStore[b.id]||{};
  const remaining=b.rows.map((r,i)=>i).filter(i=>!(i in done));
  const limit=clamp(Number($("#qLimit").value)||10,1,30);
  const rows=remaining.map(i=>b.rows[i]);
  const ranked=rankForReview(rows,limit);
  S.queue=ranked.map(x=>Object.assign({},rows[x.position],{localIndex:remaining[x.position],priority:x.priority}))}

/* ================= РЕНДЕР ================= */
function bars(el,items,colorFn){
  const max=Math.max(1,...items.map(i=>i[1]));
  el.innerHTML=items.map(([label,value,extra])=>
    '<li><span title="'+esc(label)+'">'+esc(RU[label]||label)+'</span>'+
    '<span class="track"><i style="width:'+(value/max*100)+'%;background:'+(colorFn?colorFn(label):"#8c7b70")+'"></i></span>'+
    '<strong>'+(extra!==undefined?extra:value)+'</strong></li>').join("")}

function renderKpis(){
  const b=currentBatch(),d=S.drift;
  const bm=S.benchmark;
  const winner=bm?bm.winner:null;
  const wsum=bm&&winner?bm.summary[winner]:null;
  const q=qualityData();
  const items=[
    ["Записей в датасете",S.rows.length||"—","UCI Steel Plates Faults"],
    ["Holdout",bm?bm.holdout_count:(S.holdout.length||"—"),"фиксированная тестовая выборка"],
    ["Лучшая модель OOF",winner?winner:"—",wsum?"macro-F1 "+f3(wsum.macro_f1):"нет benchmark.json"],
    ["Партия",b?b.rows.length:"—",b?b.label:"не выбрана"],
    ["Мониторинг",d?STATUS[d.status]:"—",d&&d.score!=null?"KS "+f3(d.score)+" / порог "+f3(d.threshold):"нужно ≥ 30 записей"],
    ["Accuracy на holdout",q?pc(q.accuracy):"—",q?MODEL_NAMES[S.qModel]:""],
    ["Macro-F1 на holdout",q?f3(q.macro_f1):"—",q?q.size+" записей":""],
    ["Подтверждено меток",Object.values(S.labelsStore).reduce((s,o)=>s+Object.keys(o).length,0),"готово к дообучению"]
  ];
  $("#kpis").innerHTML=items.map(([l,v,s])=>
    '<div class="card metric"><span>'+esc(l)+'</span><strong>'+esc(v)+'</strong><em>'+esc(s)+'</em></div>').join("")}

function renderBenchmark(){
  const bm=S.benchmark;
  if(!bm){$("#bmChart").innerHTML='<div class="empty">Загрузите analysis/benchmark.json</div>';
    $("#bmTime").innerHTML="";$("#bmTable").innerHTML="";return}
  $("#bmCaption").textContent="Out-of-fold macro-F1 на "+bm.train_count+" обучающих записях, среднее по семенам 42 / 47 / 53. Победитель: "+bm.winner+". sklearn "+bm.versions.sklearn+", xgboost "+bm.versions.xgboost+".";
  const entries=Object.entries(bm.summary).sort((a,b)=>b[1].macro_f1-a[1].macro_f1);
  const reps=bm.repetitions||[];
  const W=560,rh=42,H=entries.length*rh+16;
  let s='<svg class="chart" viewBox="0 0 '+W+' '+H+'" role="img" aria-label="Сравнение моделей">';
  entries.forEach(([name,v],i)=>{
    const y=i*rh+10,rs=reps.filter(r=>r.model===name).map(r=>r.macro_f1);
    const lo=rs.length?Math.min(...rs):v.macro_f1,hi=rs.length?Math.max(...rs):v.macro_f1;
    const x0=190,bw=W-x0-56;
    const isWin=name===bm.winner;
    s+='<text x="0" y="'+(y+13)+'" font-size="11.5" fill="'+(isWin?"var(--orange-ink)":"var(--ink)")+'" font-weight="'+(isWin?700:500)+'">'+esc(name)+'</text>';
    s+='<rect x="'+x0+'" y="'+y+'" width="'+bw+'" height="12" rx="6" fill="var(--soft)"/>';
    s+='<rect x="'+x0+'" y="'+y+'" width="'+(v.macro_f1*bw)+'" height="12" rx="6" fill="var(--orange)"><animate attributeName="width" from="0" to="'+(v.macro_f1*bw)+'" dur="0.6s" fill="freeze"/></rect>';
    s+='<rect x="'+x0+'" y="'+(y+16)+'" width="'+bw+'" height="7" rx="3.5" fill="var(--soft)"/>';
    s+='<rect x="'+x0+'" y="'+(y+16)+'" width="'+(v.accuracy*bw)+'" height="7" rx="3.5" fill="#8c7b70"/>';
    if(rs.length>1){const a=x0+lo*bw,b2=x0+hi*bw;
      s+='<line x1="'+a+'" y1="'+(y-4)+'" x2="'+b2+'" y2="'+(y-4)+'" stroke="var(--border)" stroke-width="3" stroke-linecap="round"/>';
      s+='<circle cx="'+a+'" cy="'+(y-4)+'" r="2" fill="#8c7b70"/><circle cx="'+b2+'" cy="'+(y-4)+'" r="2" fill="#8c7b70"/>'}
    s+='<text x="'+(W-4)+'" y="'+(y+11)+'" font-size="11" text-anchor="end" fill="var(--ink)" font-weight="650">'+f3(v.macro_f1)+'</text>';
    s+='<text x="'+(W-4)+'" y="'+(y+23)+'" font-size="9.5" text-anchor="end" fill="var(--muted)">'+f3(v.accuracy)+'</text>'});
  $("#bmChart").innerHTML=s+"</svg>";
  const times={};(bm.folds||[]).forEach(f=>{times[f.model]=(times[f.model]||0)+f.fit_seconds});
  const tEntries=Object.entries(times).sort((a,b)=>a[1]-b[1]);
  if(tEntries.length){const tmax=Math.max(...tEntries.map(t=>t[1]));
    $("#bmTime").innerHTML='<ul class="bars">'+tEntries.map(([n,v])=>
      '<li><span>'+esc(n)+'</span><span class="track"><i style="width:'+(v/tmax*100)+'%"></i></span><strong>'+v.toFixed(1)+'</strong></li>').join("")+'</ul>'}
  $("#bmTable").innerHTML='<thead><tr><th>Модель</th><th class="number">macro-F1</th><th class="number">accuracy</th><th class="number">сек.</th></tr></thead><tbody>'+
    entries.map(([n,v])=>'<tr><th scope="row">'+esc(n)+(n===bm.winner?' <span class="badge">winner</span>':'')+'</th><td class="number">'+f3(v.macro_f1)+'</td><td class="number">'+f3(v.accuracy)+'</td><td class="number">'+(times[n]?times[n].toFixed(1):"—")+'</td></tr>').join("")+'</tbody>'}

function qualityData(){
  if(!S.holdout.length)return null;
  const truth=S.holdout.map(h=>h.truth),pred=S.holdout.map(h=>h[S.qModel]);
  if(pred.some(p=>p===undefined))return null;
  return confusion(truth,pred)}

function renderQualityRadar(metrics){
  const container = $("#radarChart");
  if (!metrics) {
    container.textContent = "Для диаграммы нужны прогнозы на контрольной выборке.";
    return;
  }
  const center = {x:180, y:122};
  const axes = metrics.map((row, index) => {
    const angle = (2 * Math.PI * index / metrics.length) - Math.PI / 2;
    return {row, dx:Math.cos(angle), dy:Math.sin(angle)};
  });
  function point(axis, radius) {
    return [center.x + radius * axis.dx, center.y + radius * axis.dy];
  }
  function polygon(radiusFor) {
    return axes.map(axis => point(axis, radiusFor(axis)).join(",")).join(" ");
  }
  const rings = [0.25, 0.5, 0.75, 1].map(level =>
    '<polygon points="' + polygon(() => 94 * level) + '" fill="none" stroke="var(--border)"/>');
  const spokes = axes.map(axis => {
    const [x,y] = point(axis,94);
    return '<line x1="180" y1="122" x2="'+x+'" y2="'+y+'" stroke="var(--border)"/>';
  });
  const labels = axes.map(axis => {
    const [x,y] = point(axis,103);
    const anchor = Math.abs(axis.dx)<0.1 ? "middle" : axis.dx>0 ? "start" : "end";
    return '<text x="'+x+'" y="'+y+'" text-anchor="'+anchor+'" dominant-baseline="middle" font-size="11">'+esc(RU[axis.row.cls])+'</text>';
  });
  container.innerHTML = '<svg class="chart" viewBox="0 0 360 238" role="img" aria-label="Паутинная диаграмма F1 по типам дефектов. Точные значения приведены в таблице ниже.">'+
    rings.join("") + spokes.join("") + '<polygon points="'+ polygon(axis => 94 * axis.row.f1) +
    '" stroke="var(--orange-ink)" stroke-width="2" fill="var(--orange)" fill-opacity=".22"/>'+labels.join("")+'</svg>';
}

function renderQuality(){
  const q=qualityData();
  renderQualityRadar(q?.per);
  if(!q){$("#qKpis").innerHTML="";$("#cmChart").innerHTML='<div class="empty">Загрузите analysis/holdout_predictions.csv</div>';
    $("#qTable").innerHTML="";$("#mgChart").innerHTML="";$("#disputedTable").innerHTML="";$("#disputedKpi").innerHTML="";return}
  const worst=q.per.slice().sort((a,b)=>a.f1-b.f1)[0];
  $("#qKpis").innerHTML=[
    ["Верные прогнозы",pc(q.accuracy),q.size+" записей"],
    ["Macro-F1",f3(q.macro_f1),"равный вес семи категорий"],
    ["Слабая категория",RU[worst.cls],"F1 "+f3(worst.f1)+" · "+worst.support+" записей"],
    ["Ошибок",q.size-Math.round(q.accuracy*q.size),"из "+q.size]
  ].map(([l,v,s])=>'<div class="card metric"><span>'+esc(l)+'</span><strong>'+esc(v)+'</strong><em>'+esc(s)+'</em></div>').join("");
  const max=Math.max(1,...q.M.flat()),W=430,cell=(W-96)/7,H=cell*7+56;
  let s='<svg class="chart" viewBox="0 0 '+W+' '+H+'" role="img" aria-label="Матрица ошибок">';
  CLASSES.forEach((c,j)=>{s+='<text x="'+(96+j*cell+cell/2)+'" y="14" font-size="9" text-anchor="middle" fill="'+COLOR[c]+'" font-weight="700">'+c.slice(0,2).toUpperCase()+'</text>'});
  CLASSES.forEach((c,i)=>{
    s+='<text x="90" y="'+(24+i*cell+cell/2+3)+'" font-size="9.5" text-anchor="end" fill="'+COLOR[c]+'" font-weight="650">'+esc(RU[c])+'</text>';
    CLASSES.forEach((c2,j)=>{const v=q.M[i][j],a=v?clamp(.14+v/max*.86,.14,1):.05;
      const col=i===j?COLOR[c]:"#c0392b",x=96+j*cell,y=24+i*cell;
      s+='<rect x="'+(x+1.5)+'" y="'+(y+1.5)+'" width="'+(cell-3)+'" height="'+(cell-3)+'" rx="5" fill="'+col+'" opacity="'+a+'"><title>'+esc(RU[c])+' → '+esc(RU[c2])+': '+v+'</title></rect>';
      if(v)s+='<text x="'+(x+cell/2)+'" y="'+(y+cell/2+3.5)+'" font-size="10" font-weight="700" text-anchor="middle" fill="'+(a>.55?"#fff":"var(--muted)")+'">'+v+'</text>'})});
  s+='<text x="'+(W/2)+'" y="'+(H-6)+'" font-size="9" text-anchor="middle">прогноз модели →</text>';
  $("#cmChart").innerHTML=s+"</svg>";
  $("#qTable").innerHTML='<thead><tr><th>Категория</th><th class="number">F1</th><th class="number">Записей</th><th class="number wide">Precision</th><th class="number wide">Recall</th></tr></thead><tbody>'+
    q.per.map(r=>'<tr><th scope="row"><span class="cls"><i class="sw" style="background:'+COLOR[r.cls]+'"></i>'+esc(RU[r.cls])+'</span></th><td class="number">'+f3(r.f1)+'</td><td class="number">'+r.support+'</td><td class="number wide">'+f3(r.precision)+'</td><td class="number wide">'+f3(r.recall)+'</td></tr>').join("")+'</tbody>';
  renderMargin();renderDisputed()}

function renderMargin(){
  const key=S.qModel+"_margin";
  const vals=S.holdout.map(h=>Number(h[key])).filter(v=>!isNaN(v));
  if(!vals.length){$("#mgChart").innerHTML="";return}
  const bins=20,h=new Array(bins).fill(0);
  vals.forEach(v=>{h[clamp(Math.floor(v*bins),0,bins-1)]++});
  const max=Math.max(...h),W=520,H=150,bw=W/bins;
  const wrong=S.holdout.map((r,i)=>r.truth!==r[S.qModel]);
  const hw=new Array(bins).fill(0);
  vals.forEach((v,i)=>{if(wrong[i])hw[clamp(Math.floor(v*bins),0,bins-1)]++});
  let s='<svg class="chart" viewBox="0 0 '+W+' '+(H+24)+'" role="img" aria-label="Распределение margin">';
  h.forEach((v,i)=>{const x=i*bw,bh=v/max*H;
    s+='<rect x="'+(x+1)+'" y="'+(H-bh)+'" width="'+(bw-2)+'" height="'+Math.max(1,bh)+'" rx="2" fill="#8c7b70" opacity=".55"/>';
    const wh=hw[i]/max*H;
    if(wh>0)s+='<rect x="'+(x+1)+'" y="'+(H-wh)+'" width="'+(bw-2)+'" height="'+wh+'" rx="2" fill="#c0392b" opacity=".85"/>'});
  const mean=vals.reduce((a,b)=>a+b,0)/vals.length;
  s+='<line x1="'+(mean*W)+'" y1="0" x2="'+(mean*W)+'" y2="'+H+'" stroke="var(--orange)" stroke-width="2" stroke-dasharray="5 3"/>';
  s+='<text x="'+clamp(mean*W+5,0,W-90)+'" y="12" font-size="10" fill="var(--orange-ink)">среднее '+f3(mean)+'</text>';
  s+='<line x1="0" y1="'+H+'" x2="'+W+'" y2="'+H+'" stroke="var(--border)"/>';
  s+='<text x="0" y="'+(H+16)+'" font-size="9">0.0</text><text x="'+W+'" y="'+(H+16)+'" font-size="9" text-anchor="end">1.0 — уверенность</text>';
  $("#mgChart").innerHTML=s+"</svg>";
  $("#mgLegend").innerHTML='<span><i style="background:#8c7b70"></i>все записи</span><span><i style="background:#c0392b"></i>ошибочные прогнозы</span><span><i style="background:var(--orange)"></i>средний margin</span>'}

function renderDisputed(){
  const all=S.holdout.filter(h=>Number(h.top3_errors)===3);
  const same=all.filter(h=>h.xgb===h.hist&&h.hist===h.smote);
  $("#disputedKpi").innerHTML='<div class="metrics"><div class="card metric"><span>Ошиблись все три</span><strong>'+all.length+'</strong><em>из '+S.holdout.length+'</em></div>'+
    '<div class="card metric"><span>Совпал неверный ответ</span><strong>'+same.length+'</strong><em>общая причина, не случайность</em></div></div>';
  const rows=all.slice(0,12);
  $("#disputedTable").innerHTML=rows.length?'<thead><tr><th class="number">Строка</th><th>Истина</th><th>XGBoost</th><th>Hist</th><th>SMOTE</th><th class="number">margin</th></tr></thead><tbody>'+
    rows.map(h=>'<tr><td class="number">'+h.source_line+'</td>'+
      '<td><span class="cls"><i class="sw" style="background:'+COLOR[h.truth]+'"></i>'+esc(RU[h.truth]||h.truth)+'</span></td>'+
      '<td>'+esc(RU[h.xgb]||h.xgb)+'</td><td>'+esc(RU[h.hist]||h.hist)+'</td><td>'+esc(RU[h.smote]||h.smote)+'</td>'+
      '<td class="number">'+f3(Number(h.xgb_margin))+'</td></tr>').join("")+'</tbody>':'<tbody><tr><td class="empty">Спорных записей нет</td></tr></tbody>'}

function renderDrift(){
  const d=S.drift,b=currentBatch();
  const badge=$("#driftBadge");
  if(!d){badge.textContent="—";badge.className="badge grey";$("#driftKpis").innerHTML="";
    $("#ksChart").innerHTML='<div class="empty">Выберите партию</div>';$("#thrChart").innerHTML="";$("#driftMeta").innerHTML="";return}
  badge.textContent=STATUS[d.status];
  badge.className="badge "+(d.status==="shift"?"red":d.status==="stable"?"green":"yellow");
  $("#driftKpis").innerHTML=[
    ["Статус",STATUS[d.status],d.status==="shift"?"партия отличается от опорной":"в пределах нормы"],
    ["KS-статистика",d.score!=null?f3(d.score):"—",d.scores&&d.scores.length?d.scores[0].feature:""],
    ["Порог",d.threshold!=null?f3(d.threshold):"—","95-й квантиль, "+DRAWS+" розыгрышей"],
    ["Записей в пробе",d.sample_size,b?"всего в партии "+b.rows.length:""]
  ].map(([l,v,s])=>'<div class="card metric"><span>'+esc(l)+'</span><strong>'+esc(v)+'</strong><em>'+esc(s)+'</em></div>').join("");
  if(!d.scores||!d.scores.length){$("#ksChart").innerHTML='<div class="empty">Партия меньше 30 записей — мониторинг не запускается</div>';
    $("#thrChart").innerHTML="";$("#driftMeta").innerHTML="";
    $("#driftNote").textContent="Загрузите партию не менее чем из 30 записей.";return}
  const top=d.scores.slice(0,14),max=Math.max(d.threshold*1.15,...top.map(x=>x.score));
  const W=520,rh=22,H=top.length*rh+22;
  let s='<svg class="chart" viewBox="0 0 '+W+' '+H+'" role="img" aria-label="KS по признакам">';
  const tx=170+(W-170-46)*(d.threshold/max);
  s+='<line x1="'+tx+'" y1="0" x2="'+tx+'" y2="'+(H-18)+'" stroke="#c0392b" stroke-width="1.5" stroke-dasharray="4 3"/>';
  s+='<text x="'+clamp(tx+4,0,W-70)+'" y="10" font-size="9.5" fill="#962f2a">порог '+f3(d.threshold)+'</text>';
  top.forEach((x,i)=>{const y=i*rh+16,bw=W-170-46,w=x.score/max*bw;
    const bright=BRIGHTNESS.includes(x.feature);
    s+='<text x="0" y="'+(y+9)+'" font-size="10" fill="'+(bright?"var(--orange-ink)":"var(--muted)")+'" font-weight="'+(bright?700:400)+'">'+esc(x.feature)+'</text>';
    s+='<rect x="170" y="'+y+'" width="'+bw+'" height="11" rx="5.5" fill="var(--soft)"/>';
    s+='<rect x="170" y="'+y+'" width="'+w+'" height="11" rx="5.5" fill="'+(bright?"var(--orange)":"#8c7b70")+'"><animate attributeName="width" from="0" to="'+w+'" dur="0.5s" fill="freeze"/></rect>';
    s+='<text x="'+(W-4)+'" y="'+(y+9)+'" font-size="10" text-anchor="end" fill="var(--ink)">'+f3(x.score)+'</text>'});
  $("#ksChart").innerHTML=s+"</svg>";
  const sizes=Object.keys(S.baseline.thresholds).map(Number).sort((a,b)=>a-b);
  const tmax=Math.max(...sizes.map(k=>S.baseline.thresholds[k]));
  const TW=320,TH=120;
  let t='<svg class="chart" viewBox="0 0 '+TW+' '+(TH+22)+'" role="img" aria-label="Пороги по размеру партии">';
  sizes.forEach((k,i)=>{const x=10+i*(TW-20)/sizes.length,w=(TW-20)/sizes.length-8;
    const h=S.baseline.thresholds[k]/tmax*TH;
    const on=k===d.sample_size;
    t+='<rect x="'+x+'" y="'+(TH-h)+'" width="'+w+'" height="'+h+'" rx="4" fill="'+(on?"var(--orange)":"#8c7b70")+'" opacity="'+(on?1:.5)+'"/>';
    t+='<text x="'+(x+w/2)+'" y="'+(TH+14)+'" font-size="9" text-anchor="middle">'+k+'</text>';
    t+='<text x="'+(x+w/2)+'" y="'+(TH-h-4)+'" font-size="8.5" text-anchor="middle" fill="var(--muted)">'+f3(S.baseline.thresholds[k])+'</text>'});
  $("#thrChart").innerHTML=t+"</svg>";
  $("#driftMeta").innerHTML=[["Партия",b?b.label:"—"],["Записей",b?b.rows.length:0],
    ["Размер пробы",d.sample_size],["Признаков",d.scores.length],
    ["Опорная выборка",S.split?S.split.train.length:0],["Калибровочная",S.split?S.split.validation.length:0]]
    .map(([k,v])=>"<dt>"+esc(k)+"</dt><dd>"+esc(v)+"</dd>").join("");
  const brightHit=d.scores.slice(0,4).filter(x=>BRIGHTNESS.includes(x.feature)).length;
  $("#driftNote").className="status "+(d.status==="shift"?"warning":"neutral");
  $("#driftNote").innerHTML=d.status==="shift"
    ? "<strong>Сдвиг обнаружен.</strong><p>Среди четырёх ведущих признаков "+brightHit+" относятся к яркости. Проверьте освещение камеры и калибровку CV-модуля, затем разметьте очередь и запустите дообучение.</p>"
    : "<strong>Партия в пределах нормы.</strong><p>Максимальная KS-статистика не превышает порог для этого размера пробы. Мониторинг не заменяет контроль качества модели.</p>";
  renderDistributions()}

function renderDistributions(){
  const b=currentBatch();
  if(b){const c={};b.rows.forEach(r=>{c[r.pred]=(c[r.pred]||0)+1});
    const items=Object.entries(c).sort((a,b2)=>b2[1]-a[1]);
    bars($("#distBatch"),items,l=>COLOR[l]||"#8c7b70");
    $("#distCaption").textContent=b.note||"По прогнозам модели."}
  else{$("#distBatch").innerHTML="";$("#distCaption").textContent="Партия не выбрана."}
  if(S.labels.length){const c={};S.labels.forEach(l=>{c[l]=(c[l]||0)+1});
    bars($("#distUci"),Object.entries(c).sort((a,b2)=>b2[1]-a[1]),l=>COLOR[l]||"#8c7b70");
    $("#uciCaption").textContent="Фактические метки · "+S.labels.length+" дефектов."}}

function renderQueue(){
  const b=currentBatch();
  $("#queueTag").textContent=b?S.queue.length+" из "+b.rows.length:"—";
  if(!b||!S.queue.length){$("#queue").innerHTML='<div class="empty">Нет записей для проверки. Загрузите партию или снимите метки.</div>';return}
  const done=S.labelsStore[b.id]||{};
  $("#queue").innerHTML=S.queue.map((r,i)=>{
    const lab=done[r.localIndex];
    return '<button class="qitem'+(S.entry===r.localIndex?" on":"")+(lab?" done":"")+'" data-pos="'+r.localIndex+'">'+
      '<span class="rank">'+(lab?"✓":i+1)+'</span>'+
      '<span style="min-width:0"><span class="cls"><i class="sw" style="background:'+COLOR[r.pred]+'"></i>'+esc(RU[r.pred]||r.pred)+'</span>'+
      '<small class="muted">запись '+r.source_line+' · margin '+f3(r.margin)+(lab?" · метка: "+esc(RU[lab]||lab):"")+'</small></span>'+
      '<span class="badge grey">'+f3(r.priority)+'</span></button>'}).join("");
  $$("#queue .qitem").forEach(el=>el.onclick=()=>{S.entry=Number(el.dataset.pos);renderQueue();renderEntry()})}

function renderEntry(){
  const b=currentBatch();
  if(!b||S.entry===null||S.entry===undefined){$("#entry").innerHTML='<div class="empty">Выберите запись из очереди.</div>';$("#entryTag").textContent="—";return}
  const r=b.rows[S.entry];if(!r)return;
  const done=(S.labelsStore[b.id]||{})[S.entry];
  $("#entryTag").textContent="запись "+r.source_line;
  let h='<div class="row"><span class="cls" style="font-size:1rem"><i class="sw" style="background:'+COLOR[r.pred]+'"></i>'+esc(RU[r.pred]||r.pred)+'</span><span class="badge">margin '+f3(r.margin)+'</span>'+(done?'<span class="badge green">метка: '+esc(RU[done]||done)+'</span>':'')+'</div>';
  h+='<div><label for="labelPick">Категория контролёра</label><div class="row" style="margin-top:6px"><select id="labelPick" style="flex:1 1 160px">'+
    CLASSES.map(c=>'<option value="'+c+'"'+(c===(done||r.pred)?" selected":"")+'>'+esc(RU[c])+'</option>').join("")+'</select>'+
    '<button class="button primary" id="saveLabel">Сохранить метку</button></div></div>';
  if(r.truth)h+='<div class="status neutral"><strong>Метка UCI:</strong> '+esc(RU[r.truth]||r.truth)+' — используется только для контроля, модель её не видит.</div>';
  h+='<div><h3>27 характеристик</h3><dl class="data-list" style="margin-top:8px">'+
    S.features.map(f=>{const v=r.features[f];
      return '<dt>'+esc(f)+(BRIGHTNESS.includes(f)?' <span class="badge">яркость</span>':'')+'</dt><dd>'+(Number.isInteger(v)?v:f3(v))+'</dd>'}).join("")+'</dl></div>';
  $("#entry").innerHTML=h;
  $("#saveLabel").onclick=()=>{
    const val=$("#labelPick").value;
    S.labelsStore[b.id]=S.labelsStore[b.id]||{};
    S.labelsStore[b.id][S.entry]=val;persist();
    toast("Метка записи "+r.source_line+" сохранена · "+RU[val],"ok");
    S.entry=null;buildQueue();renderQueue();renderEntry();renderKpis()}}

function renderBatchSelect(){
  $("#batchSel").innerHTML=S.batches.map(b=>'<option value="'+b.id+'"'+(b.id===S.batchId?" selected":"")+'>'+esc(b.label)+' · '+b.rows.length+'</option>').join("")
    ||'<option value="">Партия не выбрана</option>'}
function renderDemo(){
  const d=S.demo;if(!d)return;
  $("#demoResult").innerHTML='<div class="metrics">'+
    [["Обнаружено сдвигов",pc(d.detection_rate),d.trials+" искусственных партий"],
     ["Ложных тревог",pc(d.false_alert_rate),d.trials+" обычных партий"],
     ["Размер партии",d.batch_size,"записей в каждой пробе"],
     ["Пар прогонов",d.trials,"обычная + сдвинутая"]]
    .map(([l,v,s])=>'<div class="card metric"><span>'+esc(l)+'</span><strong>'+esc(v)+'</strong><em>'+esc(s)+'</em></div>').join("")+'</div>'+
    '<div class="status '+(d.detection_rate>=.8?"success":"warning")+'" style="margin-top:10px"><strong>Сдвиг яркости: +1.4 · max(IQR, σ/2)</strong><p>Изменены четыре признака: '+BRIGHTNESS.join(", ")+'. Детектор ловит '+Math.round(d.detection_rate*d.trials)+' из '+d.trials+' сдвигов при '+Math.round(d.false_alert_rate*d.trials)+' ложных срабатываниях.</p></div>'}
function renderAll(){renderKpis();renderBenchmark();renderQuality();renderDrift();renderQueue()}

/* ================= API ================= */
/* ================= ЗАГРУЗКА РЕПОЗИТОРИЯ ================= */
async function loadAll(){
  S.sources={nna:"load",var:"load",bench:"load",hold:"load"};renderSources();
  try{const t=await fetchText("Faults27x7_var");
    S.names=t.split(/\r?\n/).map(x=>x.trim()).filter(Boolean);setSrc("var","ok")}
  catch(e){setSrc("var","bad");toast("Faults27x7_var недоступен — загрузите вручную","err")}
  try{const t=await fetchText("Faults.NNA");
    if(S.names.length){S.matrix=parseNNA(t,S.names);buildDataset();setSrc("nna","ok")}else setSrc("nna","bad")}
  catch(e){setSrc("nna","bad")}
  try{S.benchmark=JSON.parse(await fetchText("analysis/benchmark.json"));setSrc("bench","ok")}
  catch(e){setSrc("bench","bad")}
  try{S.holdout=csvObjects(await fetchText("analysis/holdout_predictions.csv"));setSrc("hold","ok")}
  catch(e){setSrc("hold","bad")}
  renderAll();
  if(S.rows.length){
    if(S.holdout.length){const rows=holdoutBatch(S.qModel);
      addBatch("holdout","Holdout · "+MODEL_NAMES[S.qModel],rows,
        "Реальные прогнозы из analysis/holdout_predictions.csv, признаки связаны по source_line.")}
    const payload = document.getElementById("dashboard-batch");
    if (payload) {
      const batch = JSON.parse(payload.textContent);
      const rows = batch.rows.map(p => ({
        position: p.row - 1, source_line: p.row, features: p.features,
        pred: p.class, score: p.score, margin: p.margin
      }));
      addBatch(batch.id, batch.name, rows, "Текущая партия со страницы распознавания.");
    }
    await calibrate()}
  else toast("Датасет не загружен. Откройте файлы вручную в блоке «Источники данных».","err");
  renderAll()}

function rebuildHoldout(){
  if(!S.holdout.length||!S.rows.length)return;
  const b=currentBatch();
  if(b&&b.id==="holdout"){addBatch("holdout","Holdout · "+MODEL_NAMES[S.qModel],holdoutBatch(S.qModel),
    "Реальные прогнозы из analysis/holdout_predictions.csv, признаки связаны по source_line.")}}

/* ================= СОБЫТИЯ ================= */
$("#reload").onclick=loadAll;
$("#qModel").onchange=e=>{S.qModel=e.target.value;persist();renderQuality();rebuildHoldout();renderKpis()};
$("#batchSel").onchange=e=>{S.batchId=e.target.value;S.entry=null;assessCurrent()};
$("#qLimit").onchange=()=>{buildQueue();renderQueue()};
$("#runDemo").onclick=runDriftDemo;
$("#useHoldout").onclick=()=>{if(!S.holdout.length){toast("holdout_predictions.csv не загружен","err");return}
  addBatch("holdout","Holdout · "+MODEL_NAMES[S.qModel],holdoutBatch(S.qModel),
    "Реальные прогнозы из analysis/holdout_predictions.csv, признаки связаны по source_line.");
  toast("Партия holdout загружена","ok")};
$("#useShift").onclick=()=>{
  if(!S.baseline){toast("Дождитесь калибровки порогов","err");return}
  const base=holdoutBatch(S.qModel).slice(0,128);
  if(base.length<MIN_BATCH){toast("Недостаточно записей для демо-сдвига","err");return}
  addBatch("shift","Сдвиг яркости · 128",shiftBrightness(base),
    "Признаки яркости сдвинуты по формуле simulate_brightness_shift. Прогнозы перенесены из исходной партии — сдвиг меняет только признаки, и это видно монитору, но не модели.");
  toast("Демо-партия со сдвигом яркости создана","ok")};
$("#clearLabels").onclick=()=>{const b=currentBatch();if(!b)return;
  const old=S.labelsStore[b.id];delete S.labelsStore[b.id];persist();buildQueue();renderQueue();renderEntry();renderKpis();
  toast("Метки партии сброшены","ok")};
$("#fVar").onchange=async e=>{const t=await e.target.files[0].text();
  S.names=t.split(/\r?\n/).map(x=>x.trim()).filter(Boolean);setSrc("var","ok");
  if(S.matrix){buildDataset();renderAll();calibrate()}};
$("#fNNA").onchange=async e=>{const t=await e.target.files[0].text();
  if(!S.names.length){toast("Сначала загрузите Faults27x7_var","err");return}
  S.matrix=parseNNA(t,S.names);buildDataset();setSrc("nna","ok");renderAll();await calibrate();renderAll()};
$("#fBench").onchange=async e=>{S.benchmark=JSON.parse(await e.target.files[0].text());setSrc("bench","ok");renderBenchmark();renderKpis()};
$("#fHold").onchange=async e=>{S.holdout=csvObjects(await e.target.files[0].text());setSrc("hold","ok");
  renderQuality();rebuildHoldout();renderKpis()};
$("#expPred").onclick=()=>{const b=currentBatch();if(!b){toast("Партия не выбрана","err");return}
  const head="Запись,Прогноз,Вероятность,Разрыв топ-2";
  const body=b.rows.map(r=>[r.source_line,r.pred??"",r.score!=null?f3(r.score):"",f3(r.margin??0)].join(",")).join("\n");
  download("steel-predictions.csv",csvBom(head+"\n"+body),"text/csv;charset=utf-8");toast("Прогнозы выгружены","ok")};
$("#expLabels").onclick=()=>{const out=["Партия,Запись,Прогноз,Метка контролёра"];
  Object.entries(S.labelsStore).forEach(([bid,marks])=>{const b=S.batches.find(x=>x.id===bid);
    Object.entries(marks).forEach(([pos,lab])=>{const r=b&&b.rows[Number(pos)];
      out.push([bid,r?r.source_line:pos,r?r.pred:"",lab].join(","))})});
  if(out.length===1){toast("Подтверждённых меток пока нет","err");return}
  download("steel-labels.csv",csvBom(out.join("\n")),"text/csv;charset=utf-8");toast("Метки выгружены","ok")};
$("#expKs").onclick=()=>{if(!S.drift||!S.drift.scores||!S.drift.scores.length){toast("Мониторинг не рассчитан","err");return}
  const head="Признак,KS,Порог,Статус";
  const body=S.drift.scores.map(x=>[x.feature,f3(x.score),f3(S.drift.threshold),STATUS[S.drift.status]].join(",")).join("\n");
  download("steel-ks.csv",csvBom(head+"\n"+body),"text/csv;charset=utf-8");toast("KS выгружены","ok")};
$("#expJson").onclick=()=>{const q=qualityData();
  download("stal-in-dashboard.json",JSON.stringify({
    generated:new Date().toISOString(),dataset:S.rows.length,split:S.split?{train:S.split.train.length,validation:S.split.validation.length,pool:S.split.pool.length,test:S.split.test.length}:null,
    benchmark:S.benchmark?{winner:S.benchmark.winner,summary:S.benchmark.summary}:null,
    holdout:{model:S.qModel,accuracy:q?q.accuracy:null,macro_f1:q?q.macro_f1:null,per_class:q?q.per:null},
    drift:S.drift?{status:S.drift.status,score:S.drift.score,threshold:S.drift.threshold,sample_size:S.drift.sample_size}:null,
    drift_demo:S.demo,labels_confirmed:Object.values(S.labelsStore).reduce((s,o)=>s+Object.keys(o).length,0),
    version:Number(document.getElementById("dashboard-version").value)},null,1),"application/json");
  toast("Сводка выгружена","ok")};
$$(".subnav a").forEach(a=>a.onclick=e=>{e.preventDefault();
  const el=document.querySelector(a.getAttribute("href"));if(el)el.scrollIntoView({behavior:"smooth",block:"start"})});
window.addEventListener("scroll",()=>{
  const ids=["s-connect","s-models","s-quality","s-drift","s-review","s-export"];
  let cur=ids[0];ids.forEach(id=>{const el=document.getElementById(id);
    if(el&&el.getBoundingClientRect().top<160)cur=id});
  $$(".subnav a").forEach(a=>a.classList.toggle("on",a.getAttribute("href")==="#"+cur))},{passive:true});

/* ================= СТАРТ ================= */
restore();
$("#qModel").value=S.qModel;
renderSources();renderAll();
loadAll();
