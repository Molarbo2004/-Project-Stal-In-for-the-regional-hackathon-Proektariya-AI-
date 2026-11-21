import {createRequire} from 'node:module';
import fs from 'node:fs/promises';
const {Workbook,SpreadsheetFile,FileBlob}=createRequire('C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1119-907-0_26.1119.907.0_x64__3k8sg7r9htsxt/dependencies/node/load.cjs')('@oai/artifact-tool');
const dir='outputs/stal-unit-28d730/';
const w=await SpreadsheetFile.importXlsx(await FileBlob.load(dir+'Юнит-экономика.xlsx'));
const s=w.worksheets.add('График');s.position=0;s.showGridLines=false;
s.getRange('A1:C42').format.columnWidth=20;
s.getRange('A1:G1').merge();s.getRange('A1').values=[['Экономика одной линии за 36 месяцев']];
s.getRange('A1').format.font={name:'Arial',size:16,bold:true,color:'#164958'};
s.getRange('A1').format.rowHeight=30;
s.getRange('A2:K2').merge();s.getRange('A2').values=[['Базовый сценарий. В месяце 0 оплачивается внедрение, с месяца 1 линия работает на плановом объёме.']];
s.getRange('A26:C26').values=[['Месяц','С внедрением, млн ₽','Без внедрения, млн ₽']];
s.getRange('A26:C26').format={fill:'#194957',font:{bold:true,color:'#FFFFFF'}};
for(let m=0;m<=36;m++){
const row=27+m;s.getRange('A'+row).values=[[m]];
s.getRange('B'+row).formulas=[["=('Юнит-экономика'!$B$15*A"+row+"/12-'Юнит-экономика'!$B$26)/1000000"]];
s.getRange('C'+row).values=[[0]];}
s.getRange('B27:C63').setNumberFormat('0.00');
const ch=s.charts.add('line',{title:'Накопленная выгода, млн ₽',hasLegend:true});ch.setPosition('A4','L24');
ch.title='Накопленная выгода, млн ₽';ch.hasLegend=true;ch.legend.position='top';
for(const [col,name,color,style] of [['B','Выгода от внедрения','#087B68','solid'],['C','Без внедрения (0 ₽)','#657584','dashed']]){
const t=ch.series.add(name);t.categoryFormula="'График'!$A$27:$A$63";t.formula="'График'!$"+col+"$27:$"+col+"$63";t.line={fill:color,style,width:3};}
ch.xAxis={axisType:'textAxis',title:'Месяцы после запуска',tickLabelInterval:3};ch.yAxis={numberFormatCode:'0.0',numberFormatSourceLinked:false};
w.recalculate();
const img=await w.render({sheetName:'График',range:'A1:L24',format:'png',scale:0.85});
await fs.writeFile('.codex-unit-work/line.png',new Uint8Array(await img.arrayBuffer()));
await (await SpreadsheetFile.exportXlsx(w)).save(dir+'Юнит-экономика с графиком.xlsx');

